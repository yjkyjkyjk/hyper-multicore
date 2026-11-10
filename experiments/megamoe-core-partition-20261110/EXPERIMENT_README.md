# MegaMoe Vector / Cube 分组实验执行口径

这些脚本与本轮报告发布在独立实验分支，不进入现有 PR，不修改旧性能报告。本轮已完成实机验证，结论见 [调度分组实验报告](README.md)，主性能采用额外的 20 次预热 / 100 次正式样本确认轮。

四组均采用六项融合全开和训练态：`baseline` 保持原调度；`vector` 仅启用全部 Vector 核并划分通信/计算；`cube` 仅划分 Cube 专家 up；`both` 同时启用。分组只作用于前向，反向调度保持原样。baseline 不是未融合 baseline，也不是 external shared。

## 使用

本目录包含实验入口与依赖脚本。在新建的隔离 worktree 完成候选构建，先 source 已确认的 CANN 环境，再使用已有只读 Python 环境；脚本不安装依赖，不修改 conda 环境。启动前检查空闲 NPU 与端口，替换下面的目录、卡号和端口。

```bash
bash run_core_partition.sh REPO PYTHON DEVICES MODE PHASE OUTPUT PORT_BASE [SHAPE]
```

- `DEVICES`：逗号分隔的空闲卡；代表性形状要求四卡。
- `MODE`：`baseline` / `vector` / `cube` / `both`。
- `PHASE`：`precision` / `performance` / `memory` / `profile`。
- `SHAPE`：`representative`（默认）或 `small`。
- `PORT_BASE`：torchrun 使用该端口，SHMEM 使用 `PORT_BASE + 1`，HCCL 使用 `PORT_BASE + 64` 到 `+127`。
- `OUTPUT`：每次使用新的结果目录；已有 `result.json` 时拒绝覆盖。
- 可选 `HP_EXPERIMENT_TRIAL=r0`、`HP_EXPERIMENT_DISPATCH_MODE=push`。正式性能默认 warmup 10 / measured 50；小于 30 样本明确标为 smoke。

建议依次完成各 mode 的 small precision、representative precision，再跑性能、显存和 DFX。每个 phase/mode 为新 torchrun 进程。性能按 `baseline → vector → cube → both` 完成一轮，再令 `HP_EXPERIMENT_TRIAL=r1` 反向跑一轮，固定卡号、shape、权重/输入 seed、routing、binary 和所有其他设置。

```bash
python3 summarize_core_partition.py RESULTS_ROOT
```

`RESULTS_ROOT` 应为新一轮输出目录，或本目录下的 `measurements/`，不要指向包含 `original_precision/` 的整个发布目录。该命令按 trial、shape、dispatch 分组，生成中文 Markdown 与 DFX 结构证据 JSON。发现同组同 mode 重复结果会拒绝合并，避免将旧二进制精度对照与候选 baseline 混算。

先在本目录解压 `artifacts/dfx-traces.tar.gz`，即可从已发布的原始 trace 生成逐 rank 的 baseline / both 物理核时间线与首对专家 up 放大图（PNG + SVG）：

```bash
python3 analyze_partition_dfx.py --baseline BASELINE_PROFILE_DIR --candidate BOTH_PROFILE_DIR --output DFX_FIGURES_DIR
```

目录须含 `rankN.json` 与 `rankN_trace.json`。脚本先校验单次前向、实际 compute 唯一任务数及 AIC/AIV 拆分与 runtime 队列数量一致（包含 terminate），再校验 Vector 角色和 Cube 专家组，不通过则写出诊断 JSON 并拒绝绘图。全图显示实际 20 Cube / 40 Vector 的物理 lane；零点为该次调用首个已记录前向任务，不能拿这张带 DFX 的图替代完整模块性能。

ready / completion 握手位于现有记录区间之外，脚本明确标记为未采集。分析包含逐核依赖等待、compute、trigger、未记录间隙、握手后 entry skew、各专家首个 up compute、专家对重叠，以及通信与 Vector 计算重叠。

## 测量与验收

- 代表性配置：每 rank T=4096、H=5120、I=S=1792、E=32、K=8、EP=4、BF16、均衡 routing、push 容量系数 1.5。继续使用公开自动 token 接口。
- small：T=128、H=5120、I=128、S=1792、每 rank 2 个专家、K=2；只作正确性和启动诊断。
- precision：同一候选 binary 先以原调度运行，再运行目标调度；相同权重种子 99173+rank、输入种子 30000+rank、同一 routing。比较 output、dx、dprob 和全部 routed/shared 参数梯度，连续三次要求 bitwise exact。它是调度回归，不能代替已有独立数学 oracle 精度测试。
- performance：训练态 forward 后接完整 backward，分别记录 forward/backward/full 的 NPU Event；记录同步后的完整 F+B 墙钟。主统计先算各 rank mean/median 再取 MAX，同时保留逐样本 MAX 的二级统计和每 rank raw samples。没有 optimizer step。禁止把独立 rank-max F/B 相加当作 full。
- memory：单独进程，先做有限性验证、清除验证临时量、warmup、zero_grad 和同步，再 reset allocator peak，运行三次完整 F+B。输出 absolute allocated/reserved peak、相对基线 allocated 增量、SHMEM heap，分别保存 rank-max 和集群 sum。当前脚本不声称得到独立前向或反向显存峰值。
- profile：同形状、同六开配置、同训练态，单独采集一次 F+B 内部 DFX，输出每 rank Chrome Trace。要求两种 direction 存在且 droppedRecordCount=0；它只解释实际调度和等待，不作为性能主数字。
- 需要额外核实 DFX：全部 Vector 核是否实际出现；通信核是否仅有 dispatch/combine，计算核是否仅有激活/归并；expert0/expert1 up 是否落在不同半组 Cube 核并真实重叠；每 task compute 恰好一次、零丢失；参与核数变化后的 ready/completion 与 profile buffer 容量正确。
- 元数据记录 commit、tracked diff SHA、未跟踪源码 SHA、forward/backward native object SHA、CANN、设备、环境、runtime header 和任务数。基线开关关闭应与冻结的 PR 原二进制另做一次精度与耗时校准，排除实验框架自身开销。

这些脚本通过导入仓库原有 `tests.torch._test_mega_moe` 复用形状、routing、输入、公开层构造和 schedule 记录；没有跑会随调度一起改变的 external_shared 对照路径。
