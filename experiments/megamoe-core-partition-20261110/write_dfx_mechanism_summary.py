#!/usr/bin/env python3
"""Write a Chinese explanation from the audited monitored DFX captures."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "dfx_figures"
DATA = json.loads((OUTPUT / "dfx_analysis.json").read_text())


def audit(rank, mode):
    return DATA[str(rank)][mode]["audit"]


def ms(value):
    return f"{value / 1000:.5f}"


lines = [
    "# 受控 DFX 内部流水与分核机制分析",
    "",
    "本节只解释调度机制；正式性能结论使用 `confirmation/` 中独立进程、20 次预热、100 次采样的无 DFX NPU Event 结果。这里的时间来自 `monitored_profile/baseline` 和 `monitored_profile/both` 各一次训练态前向内部记录，不能替代模块耗时或据此计算正式加速比。两个 capture 均使用提交 `54b0ea5` 的同一套二进制，baseline 是原调度，both 是 Vector 通信/计算分核与 Cube 专家对分核同时开启；前反向 permute、unpermute、shared expert 六项融合均开启，反向调度未变。",
    "",
    "形状：每 rank T=4096、H=5120、I=1792、shared I=1792、E=32、K=8、EP=4、BF16、push dispatch、固定 balanced route；910B3 每卡 20 Cube / 40 Vector，设备 0–3。所有图共用 baseline/candidate 横轴，并以各 rank 首个前向记录为零点。跨 rank 时钟没有做网络级对齐。",
    "",
    "## 实际核绑定和任务完整性",
    "",
    "每条 Compute 对应一次 `ExecuteComputeKernel` 调用。以下数量同时核对 runtime 任务队列、唯一 task ID、AIC/AIV 类型与终止任务，并非只检查重复数为零。",
    "",
    "| Rank | 模式 | 唯一前向 task | AIC / AIV task | terminate | 活跃 Cube / Vector | 丢记录 / 重复 | 核角色匹配 |",
    "|---:|---|---:|---:|---:|---:|---:|---|",
]

for rank in range(4):
    for mode in ("baseline", "both"):
        item = audit(rank, mode)
        assert item["complete"] and item["schedule_roles_match"]
        assert item["unique_forward_task_ids"] == item["expected_forward_task_count_including_terminate"] == 3721
        counts = item["actual_compute_counts"]
        lines.append(f"| {rank} | {mode} | {item['unique_forward_task_ids']} / 3721 | {counts['AIC']} / {counts['AIV']} | {item['terminate_compute_records']} | {len(item['aic_workers'])} / {len(item['aiv_workers'])} | {item['dropped_records']} / {item['duplicate_compute_records']} | 通过 |")

lines += [
    "",
    "both 的 AIV 1、3、…、39 只执行 Dispatch/Combine，AIV 0、2、…、38 执行激活、unpermute/epilogue 和控制任务。全部 40 个 Vector 均有记录，单种职责仍为 20 个 worker，并不等于每个阶段获得了双倍并行度。baseline 使用 20 个奇数 AIV 混排上述任务。",
    "",
    "全部 4 个 rank 的每个路由专家 up 都保持 20 个逻辑 task。baseline 每个专家分配给全部 20 Cube、每核 1 个 task；both 将本地偶数专家分配给 Cube 0–9，奇数专家分配给 Cube 10–19，每核顺序执行 2 个逻辑 task。每 rank 路由 up 仍为 160 个 task，没有减算或漏算。",
    "",
    "## 两个专家确实重叠，但第二个专家并未提前开始",
    "",
    "下表列出每个 rank 的第一个本地专家对。时间为 DFX 相对时间，单位 ms；首/末指所有物理核上的第一条/最后一条 Compute 边界。",
    "",
    "| Rank / 全局专家 ID | 第一个专家 baseline 首→末 | 第一个专家 both 首→末 | 第二个专家 baseline 首→末 | 第二个专家 both 首→末 | both 第二专家最大依赖 Wait | both 重叠 / 较短执行区间并集 |",
    "|---|---:|---:|---:|---:|---:|---:|",
]

pair_rows = []
for rank in range(4):
    before, after = audit(rank, "baseline"), audit(rank, "both")
    owners = sorted(before["up_experts"], key=int)[:2]
    first_b, second_b = [before["up_experts"][owner] for owner in owners]
    first_a, second_a = [after["up_experts"][owner] for owner in owners]
    overlap = after["up_expert_pair_overlap"]["/".join(owners)]
    def span(value):
        return f"{ms(value['first_compute_us'])}→{ms(value['last_compute_us'])}"
    lines.append(f"| {rank} / {'、'.join(owners)} | {span(first_b)} | {span(first_a)} | {span(second_b)} | {span(second_a)} | {ms(second_a['maximum_task_wait_us'])} | {ms(overlap['overlap_us'])} / {overlap['fraction_of_shorter_active_union']:.2%} |")
    pair_rows.append({"rank": rank, "owners": owners,
                      "second_expert_start_delay_us": second_a["first_compute_us"] - second_b["first_compute_us"],
                      "first_expert_completion_delay_us": first_a["last_compute_us"] - first_b["last_compute_us"],
                      "both_second_expert_max_wait_us": second_a["maximum_task_wait_us"],
                      "both_overlap_us": overlap["overlap_us"]})

lines += [
    "",
    "4 个 rank 都呈现相同趋势：第二个专家首次执行没有提前，第一专家却因分到的 Cube 减半而更晚完成。rank 0 的专家 1 在 1.63092 ms 才进入 Compute，其所属半组最长等待 1.02066 ms；与此同时专家 0 的完成从 1.48924 ms 延后到 2.04980 ms。两者虽出现 0.41888 ms 的真实重叠，仍不足以抵消每专家只使用 10 个物理核的串行 tile 成本。该结论来自核绑定、依赖 Wait 和执行边界，不需要假定 cache 退化。",
    "",
    "这里可直接观测的是依赖等待与任务首次执行；它不是网络包到达计数器，不能进一步精确声称某条数据的实际到达时刻。baseline 的 rank 1 存在 0.02466 ms 的小幅专家对重叠，来自不同 worker 的推进偏差，仍然是每个专家分配全部 20 核；不能把 baseline 描述为所有核之间存在绝对全局串行屏障。",
    "",
    "![rank 0 首对路由专家 up 的物理核执行与等待](rank0_up_pair_zoom.png)",
    "",
    "## Shared 提前完成未缩短最终依赖链",
    "",
    "rank 0 各阶段的第一/最后 Compute 边界如下。阶段彼此重叠，不能把行内跨度或跨核累计时间相加重建模块耗时。",
    "",
    "| 阶段 | baseline 首→末（ms） | both 首→末（ms） |",
    "|---|---:|---:|",
]

stage_names = ("Dispatch", "shared_up_proj", "shared_swiglu", "GMM1", "SwiGLU", "GMM2", "shared_down_proj", "Combine", "WeightedUnpermute")
stage_rows = []
for name in stage_names:
    values = [audit(0, mode)["stages"][name] for mode in ("baseline", "both")]
    lines.append(f"| {name} | {span(values[0])} | {span(values[1])} |")

lines += [
    "",
    "专用 Vector 计算组使 rank 0 shared_swiglu 的尾部从 4.76702 ms 提前至 0.80370 ms；但 shared_down 仍早于本 rank 最后一个 Combine task 完成。所有 rank 的这段余量均为正：",
    "",
    "| Rank | baseline shared_down 末 | baseline Combine 末 | 余量 | both shared_down 末 | both Combine 末 | 余量 |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]

for rank in range(4):
    values = []
    for mode in ("baseline", "both"):
        stages = audit(rank, mode)["stages"]
        shared = stages["shared_down_proj"]["last_compute_us"]
        combine = stages["Combine"]["last_compute_us"]
        values += [ms(shared), ms(combine), ms(combine - shared)]
        stage_rows.append({"rank": rank, "mode": mode, "shared_down_end_ms": shared / 1000,
                           "combine_end_ms": combine / 1000, "shared_before_combine_margin_ms": (combine - shared) / 1000,
                           "gmm1_end_ms": stages["GMM1"]["last_compute_us"] / 1000,
                           "gmm2_end_ms": stages["GMM2"]["last_compute_us"] / 1000,
                           "unpermute_start_ms": stages["WeightedUnpermute"]["first_compute_us"] / 1000,
                           "unpermute_end_ms": stages["WeightedUnpermute"]["last_compute_us"] / 1000})
    lines.append(f"| {rank} | " + " | ".join(values) + " |")

lines += [
    "",
    "因此不能仅因 shared_down 在 Cube 队列后段才执行，就把它判为最终 join 的瓶颈，或把优先前移它列为已经证实有效的优化。当前较清楚的机制是：shared_swiglu 提前主要增加了共享分支已有的余量；路由 GMM1 的尾部在所有 rank 延后，继而路由 GMM2 和 Combine 更晚完成，最后仍需 weighted unpermute。rank 0 GMM1 尾部 5.69966→6.63500 ms，GMM2 尾部 7.99766→8.97976 ms，Combine 尾部 10.62644→11.60476 ms。",
    "",
    "![rank 0 完整前向物理核流水](rank0_forward_timeline.png)",
    "",
    "## 依赖和计时边界",
    "",
    "- `hyper_multicore/modules/mega_moe/shared_schedule.py:132`、`:189`、`:235` 将共享 down 完成与路由 Combine 信号接到 weighted unpermute 的依赖事件；共享分支上游与 routing 独立。",
    "- 外部 `WaitDependency` 对应输入侧 Combine 信号及 shared down 的 join。`hyper_multicore/ops/hyper_mega_moe/op_kernel/worker_kernel.cpp:244` 在 `ExecuteWeightedUnpermute` 内还等待本 rank outgoing Combine 读完可复用存储，这段等待位于 DFX Compute 区间内。",
    "- 因此少数 rank 的 WeightedUnpermute Compute 可以早于该 rank 最后一个 outgoing Combine 的结束，例如 both rank 2 为 11.60608 ms 对 11.81114 ms。不能把该 Compute 起点直接解释为真正的 unpermute 数学计算开始，也不能用本地 Combine 尾部冒充远端 incoming 完成时间。",
    "- `hyper_multicore/ops/runtime/worker_kernel.h:172` 的入口 ready handshake 位于 `ProcessProfiled` 之前，`:185` 的 completion handshake 位于其后，两者不在图中。entry skew 只是 ready 后进入记录器的偏差，不是 ready handshake 耗时。",
    "- Compute 包住任务处理函数，通信 task 自然包含传输，某些计算 task 包含内部等待；累计 core-time 是所有核时间之和，不是 wall time。新增 worker 提前进入依赖等待会增加累计 wait，不能据此声称设备整体更空闲。",
    "- 本批 trace 可证明任务完整性、实际物理核分工、专家对执行区间重叠与阶段先后；没有 cache/bandwidth 硬件计数器，不能证明性能差异源于缓存或带宽争用。",
    "",
    "结合 `confirmation/` 的独立性能结果，本批 DFX 为当前配置继续保留原调度提供了机制解释。若后续再做调度探索，应先验证 readiness-aware 的 Cube 任务分配或按已就绪专家借用另一半空闲核，避免提前把半数核固定在尚未 ready 的专家队列上；这是待验证方向，不是本次已实现的收益。不能用旧批次未充分控制的反向重尾解释本次前向调度机制。",
    "",
    "## 文件与复现",
    "",
    "- 主图：`rank0_forward_timeline.png/.svg`、`rank0_up_pair_zoom.png/.svg`；rank 1–3 的同名文件也已生成。",
    "- 审计：`dfx_analysis.json` 保留全部 4 rank 的 task、绑定、stage、Wait、区间重叠和原 trace SHA256；`关键阶段.csv` 为关键阶段的可复用数字。",
    "- 输入：`../monitored_profile/{baseline,both}/rank{0,1,2,3}_trace.json`，以及对应 `rankN.json` 和设备占用监控记录。",
    "- 可复现命令：在结果目录运行 `python3 analyze_partition_dfx.py --baseline monitored_profile/baseline --candidate monitored_profile/both --output dfx_figures`，然后 `python3 write_dfx_mechanism_summary.py`。",
    "- 已人工检查 rank 0 全流水、rank 0 首专家对放大图、rank 2 全流水：标题、图例、核标签及说明可读，baseline/candidate 共用横轴，无裁切、文字重叠；SVG 可继续放大。",
    "",
]

(OUTPUT / "机制摘要.md").write_text("\n".join(lines))
(OUTPUT / "first_expert_pair_evidence.json").write_text(json.dumps(pair_rows, indent=2, ensure_ascii=False) + "\n")
with (OUTPUT / "关键阶段.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(stage_rows[0]))
    writer.writeheader()
    writer.writerows(stage_rows)
print(OUTPUT / "机制摘要.md")
