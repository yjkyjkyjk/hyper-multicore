#!/usr/bin/env python3
"""Verify that forward experiment selection cannot alter backward runtime bytes."""

import itertools
import json
import os
from pathlib import Path

from validate_vector_partition import load_host_modules

ROOT = Path(__file__).resolve().parents[2]
load_host_modules(ROOT)

from tests.ut.modules.mega_moe import test_shared_schedule as model
from hyper_multicore.scheduler.config import TaskSplitValue, vector_partition_task_count
from hyper_multicore.scheduler.runtime import serialize_runtime_config

checked = 0
for cores, transport, shared, route, rank in itertools.product(
        (20, 24), ("push", "pull"), (False, True), (False, True), range(4)):
    images = []
    for experiment in ("baseline", "vector", "cube", "both"):
        os.environ["HP_MEGA_MOE_CORE_PARTITION"] = experiment
        values = TaskSplitValue(tp=1, ep=4, seq_size=512, all_expert_num=8,
                                top_k=2, dispatch_mode=transport)
        values.local_num_tokens = (0, 1, 257, 512)[rank]
        builder = model.build_route_backward_graph if route else model.build_backward_graph
        options = {"integrated_permute": True, "integrated_unpermute": True} if route else {}
        graph = builder(values, hidden_size=5120, intermediate_size=128, num_cube_cores=cores,
                        shared_intermediate_size=256, fuse_shared_expert=shared, **options)
        graph.propagate_splits(values)
        cfg = model.build_backward_config(graph, values, rank, cores)
        assert vector_partition_task_count(cfg) is None
        if cfg.completion_event:
            assert cfg.all_event_num_triggers[cfg.completion_event] == 2 * cores
        images.append(serialize_runtime_config(cfg))
    assert all(image == images[0] for image in images[1:])
    checked += 1
result = {"backward_runtime_cases": checked, "modes_each": 4,
          "backward_images_byte_identical": True, "tensor_kernels_executed": False}
Path(__file__).with_name("host-backward-isolation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
