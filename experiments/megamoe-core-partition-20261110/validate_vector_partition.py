#!/usr/bin/env python3
"""Check actual host schedules without loading Torch or executing tensor kernels."""

from __future__ import annotations

import argparse
import ast
import ctypes
import importlib.machinery
import itertools
import json
import os
from pathlib import Path
import struct
import sys
import types
from dataclasses import dataclass
import __future__


def load_host_modules(repo):
    """Load pure host modules; defer annotations on the system Python 3.9."""
    class FutureLoader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            return compile(self.get_data(self.path), self.path, "exec",
                           flags=__future__.annotations.compiler_flag)

    class RepoFinder:
        @staticmethod
        def find_spec(fullname, path=None, target=None):
            if not fullname.startswith(("hyper_multicore.", "tests.ut.modules.mega_moe.")):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and isinstance(spec.loader, importlib.machinery.SourceFileLoader):
                spec.loader = FutureLoader(fullname, spec.origin)
            return spec

    sys.path.insert(0, str(repo))
    sys.meta_path.insert(0, RepoFinder)
    for package in (
        "hyper_multicore", "hyper_multicore.modules", "hyper_multicore.modules.mega_moe",
        "hyper_multicore.diagnostics", "hyper_multicore.diagnostics.profiler",
    ):
        module = types.ModuleType(package)
        module.__path__ = [str(repo / package.replace(".", "/"))]
        sys.modules[package] = module


def load_l1_mapping(repo, config_module):
    """Compile the real pure scheduling function, excluding Torch-dependent code."""
    path = repo / "hyper_multicore/diagnostics/dump/l1.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and node.name in ("_L1TaskLocation", "_task_locations")]
    namespace = {
        "__name__": __name__, "dataclass": dataclass,
        "TaskAiCoreType": config_module.TaskAiCoreType,
        "vector_partition_task_count": config_module.vector_partition_task_count,
        "_CORE_TYPE_NAMES": {1: "AIC", 2: "AIV", 3: "MIX"},
    }
    exec(compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace["_task_locations"]


def old_header(cfg, capacity):
    return struct.pack(
        "<10I2Q8x", cfg.task_num, cfg.num_workers, capacity, cfg.event_capacity,
        cfg.ready_event, cfg.diagnostic_mode, cfg.aic_profile_record_capacity,
        cfg.aiv_profile_record_capacity, cfg.completion_event, cfg.protocol_version,
        cfg.shared_tensor_table, cfg.shared_tiling,
    )


def check_config(cfg, vector_enabled, config_module, runtime, profile, task_locations):
    cores = cfg.num_workers // 2
    boundary = config_module.vector_partition_task_count(cfg)
    assert (boundary is not None) == vector_enabled
    blob = runtime.serialize_runtime_config(cfg)
    assert ctypes.sizeof(config_module.RuntimeConfigC) == 64
    if vector_enabled:
        assert struct.unpack_from("<2I", blob, 56) == (boundary, config_module.VECTOR_PARTITION_MAGIC)
    else:
        assert blob[:64] == old_header(cfg, runtime.runtime_config_task_capacity(cfg))
    if cfg.completion_event:
        assert cfg.all_event_num_triggers[cfg.completion_event] == cores * (3 if vector_enabled else 2)
    cubes = list(cfg.cube_task_indices[:cfg.task_index_num[0]])
    vectors = list(cfg.vector_task_indices[:cfg.task_index_num[1]])
    assert sorted(cubes + vectors) == list(range(cfg.task_num + 1)), "lost or duplicated task"
    groups = [cubes]
    if vector_enabled:
        groups.extend((vectors[:boundary], vectors[boundary:]))
        communication_types = {config_module.TaskType.TASK_SHMEM_GET_MEM,
                               config_module.TaskType.TASK_SHMEM_PUT_MEM_SIGNAL}
        assert all(cfg.all_tasks[task].task_type in communication_types for task in groups[1])
        assert all(cfg.all_tasks[task].task_type not in communication_types for task in groups[2])
        locations = task_locations(cfg)
        for index, task_id in enumerate(vectors):
            communication = index < boundary
            local = index if communication else index - boundary
            assert locations[task_id].core_id == 2 * (local % cores) + int(communication)
    else:
        groups.append(vectors)
    expected_records = max(
        sum(profile._records_for_task(cfg.all_tasks[task]) for task in queue[lane::cores])
        for queue in groups[1:] for lane in range(cores)
    )
    layout = profile._calculate_event_driven_profile_layout(cfg)
    assert layout.aiv_required_records == expected_records
    assert layout.aiv_record_capacity >= expected_records
    return groups


def simulate(configs, groups_by_rank, test_module):
    events = [[0] * cfg.event_capacity for cfg in configs]
    workers = []
    for rank, (cfg, groups) in enumerate(zip(configs, groups_by_rank)):
        if cfg.ready_event:
            events[rank][cfg.ready_event] = 1
        cores = cfg.num_workers // 2
        workers.extend([rank, queue[lane::cores], 0] for queue in groups for lane in range(cores))
    while True:
        pending = progress = 0
        for worker in workers:
            rank, queue, index = worker
            if index == len(queue):
                continue
            pending += 1
            cfg = configs[rank]
            task = cfg.all_tasks[queue[index]]
            if not test_module._dependencies_ready(task, cfg, events[rank]):
                continue
            test_module._signal_task(task, events, rank)
            worker[2] += 1
            progress += 1
        if not pending:
            return
        assert progress, "worker queues contain an event dependency cycle"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    load_host_modules(args.repo)
    from tests.ut.modules.mega_moe import test_shared_schedule as model
    from hyper_multicore.scheduler import config, runtime
    from hyper_multicore.diagnostics.profiler import cycle_trace as profile
    from hyper_multicore.modules.mega_moe.vector_partition_experiment import partition_vector_workers
    locations = load_l1_mapping(args.repo, config)
    topologies = (
        (1, 2, 1, 128, (1,)),
        (2, 4, 2, 512, (128, 257)),
        (2, 6, 2, 512, (0, 257)),
        (4, 8, 2, 512, (0, 1, 257, 512)),
        (4, 32, 8, 4096, (4096, 4096, 4096, 4096)),
    )
    summary = {"passed": 0, "expected_cube_rejections": 0, "mode_counts": {},
               "ranks_checked": 0, "tensor_kernels_executed": False}
    matrix = itertools.product(
        topologies, (1, 20, 24), ("push", "pull"),
        (model.build_forward_graph, model.build_route_forward_graph),
        (False, True), (False, True), ("baseline", "vector", "cube", "both"),
    )
    for topology, cores, transport, builder, shared, unpermute, mode in matrix:
        ep, experts, top_k, seq, lengths = topology
        os.environ["HP_MEGA_MOE_CORE_PARTITION"] = mode
        configs, groups = [], []
        try:
            for rank, tokens in enumerate(lengths):
                values = config.TaskSplitValue(tp=1, ep=ep, seq_size=seq,
                                               all_expert_num=experts, top_k=top_k, dispatch_mode=transport)
                values.local_num_tokens = tokens
                graph = builder(values, hidden_size=5120, intermediate_size=128,
                                num_cube_cores=cores, integrated_unpermute=unpermute,
                                shared_intermediate_size=256, fuse_shared_expert=shared)
                graph.propagate_splits(values)
                cfg = model.build_config_for_rank(graph, values, rank, cores)
                configs.append(cfg)
                groups.append(check_config(cfg, mode in ("vector", "both"),
                                           config, runtime, profile, locations))
                if mode in ("vector", "both"):
                    original = runtime.serialize_runtime_config(cfg)
                    partition_vector_workers(cfg)
                    assert runtime.serialize_runtime_config(cfg) == original, "partition is not idempotent"
        except ValueError as error:
            if cores == 1 and mode in ("cube", "both") and "positive even Cube count" in str(error):
                summary["expected_cube_rejections"] += 1
                continue
            print({"topology": topology, "cores": cores, "transport": transport,
                   "builder": builder.__name__, "shared": shared, "unpermute": unpermute, "mode": mode})
            raise
        simulate(configs, groups, model)
        summary["passed"] += 1
        summary["ranks_checked"] += len(configs)
        summary["mode_counts"][mode] = summary["mode_counts"].get(mode, 0) + 1
    summary["notes"] = [
        "Uses actual host graph builders, serializer, profile sizing and pure L1 location function.",
        "No Torch stub or tensor kernel execution; package initializers are bypassed.",
        "Python 3.9 uses postponed annotation compilation without editing repository source.",
        "Checks queue coverage, event progress, per-worker DFX capacity, L1 physical IDs, header bytes and completion.",
    ]
    result = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
