# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experimental two-expert scheduling over equal Cube worker groups."""

from hyper_multicore.scheduler.config import RuntimeConfigC, TaskType


def partition_cube_up_workers(cfg: RuntimeConfigC) -> None:
    """Assign each pair of routed up projections to half the Cube workers.

    Args:
        cfg: Forward runtime image with its existing Cube task queue.

    Raises:
        ValueError: Core count or the routed up task span is inconsistent.
    """
    num_cube_cores = cfg.num_workers // 2
    if num_cube_cores <= 0 or num_cube_cores % 2:
        raise ValueError("paired up projections require a positive even Cube count")
    queue = list(cfg.cube_task_indices[:cfg.task_index_num[0]])
    positions = [
        index for index, task_id in enumerate(queue)
        if cfg.all_tasks[task_id].task_type == TaskType.TASK_GROUPED_MATMUL
        and cfg.all_tasks[task_id].tiling_data_position == 17
        and cfg.all_tasks[task_id].inputs[0].tensor_type != 2
    ]
    if not positions:
        return
    start, count = positions[0], len(positions)
    tasks = queue[start:start + count]
    if (start % num_cube_cores or count % num_cube_cores
            or positions != list(range(start, start + count))
            or [cfg.all_tasks[task_id].task_index for task_id in tasks] != list(range(count))):
        raise ValueError("routed up projections must occupy aligned expert-major Cube rounds")
    half = num_cube_cores // 2
    paired_count = count // (2 * num_cube_cores) * (2 * num_cube_cores)
    for pair in range(0, paired_count, 2 * num_cube_cores):
        reordered = []
        for lane_start in (0, half):
            for expert_start in (pair, pair + num_cube_cores):
                reordered.extend(tasks[expert_start + lane_start:expert_start + lane_start + half])
        cfg.cube_task_indices[start + pair:start + pair + 2 * num_cube_cores] = reordered
