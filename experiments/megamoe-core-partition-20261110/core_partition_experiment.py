#!/usr/bin/env python3
"""Local-only, four-mode MegaMoe scheduling experiment worker for torchrun."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace


MODES = ("baseline", "vector", "cube", "both")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--phase", choices=("precision", "performance", "memory", "profile"), required=True)
    parser.add_argument("--shape", choices=("representative", "small"), default="representative")
    parser.add_argument("--dispatch-mode", choices=("push", "pull"), default="push")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measured", type=int, default=50)
    parser.add_argument("--precision-repeats", type=int, default=3)
    parser.add_argument("--trial", default="r0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.measured <= 0 or args.precision_repeats <= 0:
        parser.error("warmup must be nonnegative; measured and precision-repeats must be positive")
    return args


def initialize(args):
    os.environ.update({
        "HP_MEGA_MOE_CORE_PARTITION": args.mode,
        "HP_MEGA_MOE_WORLD_SIZE": os.environ["WORLD_SIZE"],
        "HP_MEGA_MOE_DISPATCH_MODE": args.dispatch_mode,
        "HP_MEGA_MOE_SHARED_INTERMEDIATE_SIZE": "1792",
        "HP_MEGA_MOE_FORWARD_SHARED_EXPERT": "1",
        "HP_MEGA_MOE_BACKWARD_SHARED_EXPERT": "1",
        "HYPER_MULTICORE_MEGA_MOE_FORWARD_FUSION_MODE": "both",
        "HYPER_MULTICORE_MEGA_MOE_BACKWARD_FUSION_MODE": "both",
    })
    import torch
    import torch.distributed as dist
    from tests.torch import _test_mega_moe as common
    from hyper_multicore import shmem

    world = dist.get_world_size()
    if args.shape == "representative":
        if world != 4:
            raise ValueError("the representative fixed shape requires four ranks")
        shape = common.performance_shape()
    else:
        shape = common.MoeShape(local_num_tokens=128, hidden_size=5120,
                                intermediate_size=128, num_experts=2 * world,
                                top_k=2, ep_size=world)
    ids, probabilities, counts = common.make_balanced_route(shape)
    hidden, upstream = common.make_data(shape)
    args.output.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(torch=torch, dist=dist, common=common, shmem=shmem,
                           shape=shape, device=common.DEVICE, rank=common.RANK,
                           hidden=hidden, upstream=upstream, ids=ids,
                           probabilities=probabilities, counts=counts)


def new_layer(context, mode):
    os.environ["HP_MEGA_MOE_CORE_PARTITION"] = mode
    context.torch.manual_seed(99173 + context.rank)
    context.torch.npu.manual_seed(99173 + context.rank)
    return context.common._new_performance_layer(context.shape, "mega_moe")


def train_step(context, layer, *, timed=False, capture=False):
    torch = context.torch
    layer.zero_grad(set_to_none=True)
    hidden = context.hidden.detach().requires_grad_(True)
    probabilities = context.probabilities.detach().requires_grad_(True)
    context.dist.barrier()
    torch.npu.synchronize(context.device)
    if timed:
        start, middle, end = (torch.npu.Event(enable_timing=True) for _ in range(3))
        wall_start = time.perf_counter()
        start.record()
    output = context.common.forward_layer(layer, hidden, context.ids, probabilities,
                                          tokens_per_expert=context.counts)
    if timed:
        middle.record()
    output.backward(context.upstream)
    if timed:
        end.record()
    torch.npu.synchronize(context.device)
    result = {}
    if timed:
        result = {"forward_event_ms": start.elapsed_time(middle),
                  "backward_event_ms": middle.elapsed_time(end),
                  "full_event_ms": start.elapsed_time(end),
                  "full_sync_wall_ms": (time.perf_counter() - wall_start) * 1000}
    if capture:
        tensors = {"output": output, "input_grad": hidden.grad, "route_grad": probabilities.grad}
        tensors.update({f"{name}_grad": value.grad for name, value in layer.named_parameters()})
        if any(value is None for value in tensors.values()):
            raise AssertionError("a required output or gradient is missing")
        result = {name: value.detach().cpu() for name, value in tensors.items()}
    return result


def schedule_evidence(context, layer):
    from hyper_multicore.scheduler.runtime import serialize_runtime_config

    result = {"shared_expert": {}}
    context.common._record_performance_schedule(result, layer, "mega_moe", context.shape)
    plan = layer._resource_group.resources.plan_for_tokens(context.shape.local_num_tokens, context.device)
    _, forward, _, backward = context.common._build_runtime_artifacts(plan.spec)
    for direction, config in (("forward", forward), ("backward", backward)):
        blob = serialize_runtime_config(config)
        result["schedule"][direction].update({
            "runtime_bytes": len(blob), "runtime_sha256": hashlib.sha256(blob).hexdigest(),
            "header_hex": blob[:64].hex(), "task_index_counts": list(config.task_index_num),
            "num_workers": int(config.num_workers),
        })
    return result


def precision(context, args):
    reference_layer = new_layer(context, "baseline")
    try:
        expected = train_step(context, reference_layer, capture=True)
    finally:
        reference_layer.close()
    del reference_layer
    gc.collect()
    context.torch.npu.empty_cache()
    context.dist.barrier()
    candidate = new_layer(context, args.mode)
    cases = []
    try:
        for repeat in range(args.precision_repeats):
            actual = train_step(context, candidate, capture=True)
            if actual.keys() != expected.keys():
                raise AssertionError("baseline/candidate gradient inventories differ")
            tensors = {}
            for name, observed in actual.items():
                reference = expected[name]
                finite = bool(context.torch.isfinite(observed).all())
                exact = bool(context.torch.equal(observed, reference))
                difference = (observed.float() - reference.float()).abs()
                observed_bytes = observed.contiguous().view(context.torch.uint8).numpy().tobytes()
                reference_bytes = reference.contiguous().view(context.torch.uint8).numpy().tobytes()
                tensors[name] = {"exact": exact, "finite": finite,
                                 "max_abs": float(difference.max()) if difference.numel() else 0.0,
                                 "shape": list(observed.shape),
                                 "sha256": hashlib.sha256(observed_bytes).hexdigest(),
                                 "expected_sha256": hashlib.sha256(reference_bytes).hexdigest()}
            cases.append({"repeat": repeat, "tensors": tensors})
            del actual
        schedule = schedule_evidence(context, candidate)
    finally:
        candidate.close()
    return {"passed": all(v["exact"] and v["finite"] for case in cases for v in case["tensors"].values()),
            "reference": "same candidate binary, six fusion switches on, scheduling mode baseline",
            "tolerance": {"rtol": 0, "atol": 0}, "cases": cases, **schedule}


def statistics_for(values):
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "minimum": min(values), "maximum": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}


def prepare_measured(context, layer, warmup):
    validation = train_step(context, layer, capture=True)
    if not all(bool(context.torch.isfinite(tensor).all()) for tensor in validation.values()):
        raise AssertionError("non-finite output or gradient before measurement")
    del validation
    layer.zero_grad(set_to_none=True)
    gc.collect()
    context.torch.npu.synchronize(context.device)
    context.torch.npu.empty_cache()
    for _ in range(warmup):
        train_step(context, layer)
    layer.zero_grad(set_to_none=True)
    gc.collect()
    context.torch.npu.synchronize(context.device)


def performance(context, args, layer):
    prepare_measured(context, layer, args.warmup)
    measurement_start_ns = time.time_ns()
    samples = [train_step(context, layer, timed=True) for _ in range(args.measured)]
    measurement_end_ns = time.time_ns()
    return {"passed": True, "samples": samples,
            "measurement_start_ns": measurement_start_ns, "measurement_end_ns": measurement_end_ns,
            "statistics": {key: statistics_for([sample[key] for sample in samples]) for key in samples[0]},
            "smoke_only": args.measured < 30, **schedule_evidence(context, layer)}


def memory(context, args, layer):
    prepare_measured(context, layer, args.warmup)
    torch = context.torch
    baseline_allocated = torch.npu.memory_allocated(context.device)
    baseline_reserved = torch.npu.memory_reserved(context.device)
    torch.npu.reset_peak_memory_stats(context.device)
    for _ in range(3):
        train_step(context, layer)
    peak_allocated = torch.npu.max_memory_allocated(context.device)
    values = {"baseline_allocated": baseline_allocated, "baseline_reserved": baseline_reserved,
              "peak_allocated": peak_allocated, "peak_reserved": torch.npu.max_memory_reserved(context.device),
              "peak_allocated_increment": peak_allocated - baseline_allocated,
              "current_allocated": torch.npu.memory_allocated(context.device),
              "current_reserved": torch.npu.memory_reserved(context.device),
              "shmem_heap": (context.shmem.debug_state().get("config") or {}).get("heap_size_bytes", 0)}
    return {"passed": True, "memory_bytes": values,
            "scope": "steady-state full training forward+backward; reset after validation+warmup+zero_grad",
            "iterations": 3, "separate_forward_backward_peaks": False,
            **schedule_evidence(context, layer)}


def profile(context, args, layer):
    from hyper_multicore.diagnostics import profiler as profiler_api, schedule

    prepare_measured(context, layer, args.warmup)
    with profiler_api.mega_kernel_profile(schedule=schedule(wait=0, warmup=0, active=1, repeat=1),
                                         detailed_task_names=True) as profiler:
        train_step(context, layer)
        profiler.step()
    path = args.output / f"rank{context.rank}_trace.json"
    trace = profiler.export_chrome_trace(path)
    metadata = trace["megaKernelCycleTrace"]
    directions = {event.get("args", {}).get("direction") for event in trace["traceEvents"]
                  if event.get("cat") == "MegaKernelInternal"}
    passed = metadata["droppedRecordCount"] == 0 and directions == {"forward", "backward"}
    return {"passed": passed, "trace": str(path), "metadata": metadata,
            "instrumented_not_a_latency_measurement": True, **schedule_evidence(context, layer)}


def provenance(context, args):
    repo = Path(context.common.__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    patch = subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)
    objects = {}
    kernel_root = repo / "build/native/payload/hyper_multicore/lib/vendors/hyper_multicore_nn/op_impl/ai_core/tbe/kernel"
    for op in ("hyper_mega_moe", "hyper_mega_moe_grad"):
        for path in sorted(kernel_root.glob(f"*/{op}/*.o")):
            objects[str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=repo, text=True)
    sources = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
               for name in untracked.splitlines()
               if Path(name).suffix in {".py", ".h", ".hpp", ".cpp", ".patch"} and (repo / name).is_file()}
    return {"repo": str(repo), "commit": commit, "tracked_diff_sha256": hashlib.sha256(patch).hexdigest(),
            "pid": os.getpid(), "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "untracked_source_sha256": sources,
            "native_objects": objects, "python": os.sys.executable,
            "torch": context.torch.__version__, "device": context.torch.npu.get_device_name(context.device),
            "device_limits": context.torch.npu.get_device_limit(context.device),
            "cann_root": os.getenv("ASCEND_HOME_PATH"), "vendor": os.getenv("ASCEND_CUSTOM_OPP_PATH"),
            "visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
            "shape": asdict(context.shape), "shared_intermediate_size": 1792,
            "dtype": "bfloat16", "dispatch_mode": args.dispatch_mode,
            "all_six_fusion_switches": True, "training": True, "backward_schedule_changed": False,
            "input_seed": 30000 + context.rank, "weight_seed": 99173 + context.rank,
            "route": "make_balanced_route", "capacity_factor": 1.5 if args.dispatch_mode == "push" else None}


def aggregate(ranks, args):
    result = {"mode": args.mode, "phase": args.phase, "trial": args.trial, "ranks": ranks,
              "passed": all(rank["result"]["passed"] for rank in ranks)}
    if args.phase == "performance":
        keys = ranks[0]["result"]["samples"][0]
        result["rank_max_statistics"] = {
            key: {stat: max(rank["result"]["statistics"][key][stat] for rank in ranks)
                  for stat in ("mean", "median", "minimum", "maximum", "stdev")} for key in keys}
        result["samplewise_rank_max"] = {}
        for key in keys:
            samples = [max(rank["result"]["samples"][i][key] for rank in ranks)
                       for i in range(args.measured)]
            result["samplewise_rank_max"][key] = {"samples": samples, **statistics_for(samples)}
        result["aggregation"] = "primary: per-rank statistics then MAX; secondary: per-sample MAX then statistics"
    elif args.phase == "memory":
        keys = ranks[0]["result"]["memory_bytes"]
        result["rank_max_memory_bytes"] = {key: max(rank["result"]["memory_bytes"][key] for rank in ranks)
                                           for key in keys}
        result["sum_memory_bytes"] = {key: sum(rank["result"]["memory_bytes"][key] for rank in ranks)
                                      for key in keys}
    return result


def main():
    args = arguments()
    context = initialize(args)
    if args.phase == "precision":
        result = precision(context, args)
    else:
        layer = new_layer(context, args.mode)
        try:
            result = {"performance": performance, "memory": memory, "profile": profile}[args.phase](context, args, layer)
        finally:
            layer.close()
    record = {"mode": args.mode, "phase": args.phase, "trial": args.trial, "rank": context.rank,
              "warmup": args.warmup, "measured": args.measured,
              "provenance": provenance(context, args), "result": result}
    path = args.output / f"rank{context.rank}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    context.dist.barrier()
    if context.rank == 0:
        ranks = [json.loads((args.output / f"rank{rank}.json").read_text())
                 for rank in range(context.dist.get_world_size())]
        summary = aggregate(ranks, args)
        (args.output / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"mode": args.mode, "phase": args.phase, "passed": summary["passed"],
                          "output": str(args.output / "result.json")}), flush=True)
    context.dist.barrier()
    passed = json.loads((args.output / "result.json").read_text())["passed"]
    context.dist.destroy_process_group()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
