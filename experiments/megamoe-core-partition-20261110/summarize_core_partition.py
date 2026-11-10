#!/usr/bin/env python3
"""Summarize completed local scheduling experiments without mixing shapes or trials."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


MODES = ("baseline", "vector", "cube", "both")
LABELS = {"baseline": "原调度（六开）", "vector": "仅 Vector 分组", "cube": "仅 Cube 分组", "both": "Vector + Cube 分组"}


def group_key(result):
    provenance = result["ranks"][0]["provenance"]
    return (result.get("trial", "r0"), json.dumps(provenance["shape"], sort_keys=True), provenance["dispatch_mode"])


def merge_intervals(intervals):
    merged = []
    for begin, end in sorted(intervals):
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])
    return merged


def overlap(first, second):
    first, second = merge_intervals(first), merge_intervals(second)
    total = 0.0
    i = j = 0
    while i < len(first) and j < len(second):
        total += max(0.0, min(first[i][1], second[j][1]) - max(first[i][0], second[j][0]))
        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1
    shorter = min(sum(end - begin for begin, end in first), sum(end - begin for begin, end in second))
    return {"overlap_us": total, "fraction_of_shorter_active_union": total / shorter if shorter else None}


def profile_evidence(path, rank_record=None):
    from analyze_partition_dfx import audit_trace
    return audit_trace(json.loads(path.read_text()), rank_record)


def markdown(groups, profiles):
    lines = ["# MegaMoe Vector / Cube 分组调度实验", "",
             "本轮 baseline 是 PR 提交 `45599d6` 的六项融合全开路径，仅保持原调度；candidate 仅改变前向核分配，反向调度不变。不是原始未融合路径与全融合路径的比较。", "",
             "训练态，包含输出、输入梯度、路由权重梯度、路由专家及共享专家权重梯度。主指标为每 rank 各自统计后取最大值的 NPU Event 耗时；原始每 rank 样本及逐样本 rank-max 统计均保留。完整 F+B 使用独立起止 event，不能把独立 rank-max 前向和反向相加。", "",
             "显存为独立进程中的稳态完整 F+B allocator 峰值，验证和 warmup 完成并清空参数梯度后重置 peak。它不是独立前向/反向峰值；SHMEM 不属于 PyTorch allocator。DFX 单独采集，不用带 instrumentation 的耗时评价性能。", ""]
    for (trial, shape_json, dispatch), phases in sorted(groups.items()):
        shape = json.loads(shape_json)
        lines.extend([f"## {trial} / {dispatch} / T={shape['local_num_tokens']} H={shape['hidden_size']} I={shape['intermediate_size']} E={shape['num_experts']} K={shape['top_k']} EP={shape['ep_size']}", ""])
        performances = phases.get("performance", {})
        memories = phases.get("memory", {})
        precisions = phases.get("precision", {})
        baseline = performances.get("baseline")
        baseline_ms = baseline["rank_max_statistics"]["full_event_ms"]["median"] if baseline else None
        baseline_forward = baseline["rank_max_statistics"]["forward_event_ms"]["median"] if baseline else None
        baseline_mem = memories.get("baseline", {}).get("rank_max_memory_bytes", {}).get("peak_allocated")
        lines.extend(["| 配置 | 精度 | F median ms | B median ms | F+B median ms | F+B mean ms | 前向耗时下降 | F+B 耗时下降 | Allocated 峰值 MiB | Reserved 峰值 MiB | Allocated 峰值下降 |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for mode in MODES:
            perf = performances.get(mode)
            mem = memories.get(mode, {}).get("rank_max_memory_bytes", {})
            precision = precisions.get(mode)
            verified = "通过（精确一致）" if precision and precision["passed"] else "失败" if precision else "未提供"
            if perf:
                stats = perf["rank_max_statistics"]
                values = [f"{stats[key]['median']:.4f}" for key in ("forward_event_ms", "backward_event_ms", "full_event_ms")]
                values.append(f"{stats['full_event_ms']['mean']:.4f}")
                change = f"{(1 - stats['full_event_ms']['median'] / baseline_ms) * 100:.2f}%" if baseline_ms else "—"
                forward_change = f"{(1 - stats['forward_event_ms']['median'] / baseline_forward) * 100:.2f}%" if baseline_forward else "—"
            else:
                values, change, forward_change = ["—"] * 4, "—", "—"
            allocated = f"{mem['peak_allocated'] / 2**20:.2f}" if mem else "—"
            reserved = f"{mem['peak_reserved'] / 2**20:.2f}" if mem else "—"
            saved = f"{(1 - mem['peak_allocated'] / baseline_mem) * 100:.2f}%" if mem and baseline_mem else "—"
            lines.append("| " + " | ".join([LABELS[mode], verified, *values, forward_change, change, allocated, reserved, saved]) + " |")
        lines.extend(["", "各 mode 的二进制 SHA、源码提交与 diff SHA、实际运行时任务数和 header、设备、环境、采样次数详见原始 result.json。负数表示退化。", ""])
    if profiles:
        lines.extend(["## DFX 证据", "", "| 捕获 | AIC 核数 | AIV 核数 | compute / 应有任务数 | 完整 | 丢失记录 | 通信与 Vector 计算重叠 µs |", "|---|---:|---:|---:|---|---:|---:|"])
        for name, evidence in sorted(profiles.items()):
            complete = "是" if evidence["complete"] and evidence["schedule_roles_match"] else "否"
            counts = f"{evidence['forward_compute_records']} / {evidence['expected_forward_task_count_including_terminate']}"
            lines.append(f"| {name} | {len(evidence['aic_workers'])} | {len(evidence['aiv_workers'])} | {counts} | {complete} | {evidence['dropped_records']} | {evidence['communication_vector_compute_overlap']['overlap_us']:.3f} |")
        lines.extend(["", "每个 AIV 的任务类别、每个专家 up 的物理核集合、专家对的 compute 区间重叠，以及逐核等待/计算/trigger 分解见配套 JSON。重叠以各类别区间并集的交集计算，不累加跨核重复时间。", "",
                      "现有 DFX 在 WaitForDeviceReady 之后才初始化记录器，CompleteDeviceReads 又在记录结束之后执行，因此不包含 entry ready/completion 握手时长。core entry spread 仅为握手后的入场偏差；任务 WaitDependency 不能冒充 ready 握手。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    groups = defaultdict(lambda: defaultdict(dict))
    profiles = {}
    paths = []
    for path in sorted(args.root.rglob("result.json")):
        result = json.loads(path.read_text())
        if result.get("mode") not in MODES or not result.get("ranks"):
            continue
        key, phase, mode = group_key(result), result["phase"], result["mode"]
        if mode in groups[key][phase]:
            raise ValueError(f"duplicate trial/shape/dispatch/phase/mode: {path}; use different --trial values")
        groups[key][phase][mode] = result
        paths.append(str(path))
        if phase == "profile":
            for rank in result["ranks"]:
                trace = path.parent / f"rank{rank['rank']}_trace.json"
                if trace.is_file():
                    profiles[str(trace.relative_to(args.root))] = profile_evidence(trace, rank)
    if not paths:
        raise SystemExit("No completed experiment result.json files found; no performance claim generated.")
    output = args.output or args.root / "调度分组实验汇总.md"
    output.write_text(markdown(groups, profiles), encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps({"result_files": paths, "profiles": profiles},
                                                     ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
