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
"""Forward-only experiment assigning communication and compute separate AIVs."""

from hyper_multicore.scheduler.config import (
    RuntimeConfigC, TaskType, VECTOR_PARTITION_MAGIC, vector_partition_task_count,
)


def partition_vector_workers(cfg: RuntimeConfigC) -> None:
    """Keep task order within odd-AIV communication and even-AIV computation."""
    if cfg.num_workers <= 0 or cfg.num_workers % 2 or cfg.task_index_num[2]:
        raise ValueError("Vector partition requires paired workers without MIX tasks")
    if cfg._padding[1] not in (0, VECTOR_PARTITION_MAGIC):
        raise ValueError("Runtime header already contains another scheduling experiment")
    communication_types = {TaskType.TASK_SHMEM_GET_MEM, TaskType.TASK_SHMEM_PUT_MEM_SIGNAL}
    compute_types = {TaskType.TASK_SWI_GLU, TaskType.TASK_WEIGHTED_UNPERMUTE, TaskType.TASK_TERMINATE}
    communication, compute = [], []
    task_ids = list(cfg.vector_task_indices[:cfg.task_index_num[1]])
    for task_id in task_ids:
        kind = cfg.all_tasks[task_id].task_type
        if kind not in communication_types | compute_types:
            raise ValueError(f"Unsupported forward Vector task type {kind}")
        (communication if kind in communication_types else compute).append(task_id)
    if not compute or cfg.all_tasks[compute[-1]].task_type != TaskType.TASK_TERMINATE:
        raise ValueError("Forward Vector queue must end with termination")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Forward Vector queue contains duplicate tasks")
    cfg.vector_task_indices[:len(task_ids)] = communication + compute
    cfg._padding[:] = (len(communication), VECTOR_PARTITION_MAGIC)
    if cfg.completion_event:
        cfg.all_event_num_triggers[cfg.completion_event] = 3 * cfg.num_workers // 2
    vector_partition_task_count(cfg)
