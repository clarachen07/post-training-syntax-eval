# EWT 冻结模型行为评测：最终报告

**状态：已完成。** 四个官方 checkpoint 各完成原文版 P0 与开发集选优版 P3 的 EWT r2.15 全量 test；八个单元各有 2,077 个唯一句子、25,094 个 gold 词，逐句推理状态均为 `ok`。所有格式错误、提前停止和预算截断均保留并计错；没有任务微调、LoRA、量化或重试择优。运行配置、权重核验和原始输出已在原实验目录保存；公开结果快照的范围见下文。

## 问题与协议

本实验测量 Llama-3.1-8B Base → Tülu-3-8B SFT → DPO → Final/RLVR 的**可观察依存分析行为**。P0 固定 Matsuda et al. (2025) [论文](https://aclanthology.org/2025.iwpt-1.2/)的三步 prompt、输入转换和作者评分代码（源码 commit `f3757d250efc9880ba77651ec0d8587cfd673477`）。输入包含原句和 gold 分词后的 FORM；每句一次生成 Task 1 UPOS、Task 2 HEAD、Task 3 DEPREL 三表。仅使用整数 ID 词，跳过 multiword token 与 empty node。test 目标 gold 标签和边不进入推理上下文。

数据是 UD English EWT r2.15，固定 commit `4dc8e10cf32352e11ab2c46e024b19853b91546e`。四模型 checkpoint revision 与只读路径见 [`experiment.json`](../experiment.json)，原始运行配置（已去除私人模型路径）见 [`run-config.json`](provenance/run-config.json)。所有权重分片的 SHA-256 及配置/tokenizer 核验见 [`weights.json`](provenance/weights.json)。Base 使用作者文本外壳；三款 Tülu 模型使用原生模板。推理统一采用 Python 3.11、PyTorch 2.5.1、Transformers 4.49.0、vLLM 0.7.2、未量化 BF16、greedy、temperature 0、top-p 1、禁用 top-k、repetition penalty 1、seed 42、8192 token 总上下文和输入后剩余全部输出预算；单卡 RTX 3090 24 GB，最大并发 4。完整环境及文件哈希见 [`frozen.json`](provenance/frozen.json)和 [`requirements-lock.txt`](provenance/requirements-lock.txt)。

严格口径以固定 gold 词数为分母，只对唯一合法 ID 与正确 FORM 对齐的词评分；缺失、重复、非法及歧义输出记错，不修复预测边。主 LAS 比较去除 DEPREL 子类型，完整子类型另报。主表包含标点；逐模型 JSON 另含不计 gold `PUNCT` 的结果。作者兼容口径沿用其最终表 F1 与恢复规则；Task 1 的兼容 F1 是本实验扩展，并非论文原生指标。

## 原文版 P0：主实验

| 阶段 | Task 1 严格 UPOS | Task 1 兼容 F1 扩展 | 最终表作者 UPOS F1 | 严格 UAS | 严格 LAS | 作者 UAS F1 | 作者 LAS F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1.31% | 2.34% | 3.46% | 0.12% | 0.03% | 0.93% | 0.57% |
| SFT | 45.30% | 56.16% | 30.03% | 4.36% | 2.47% | 7.34% | 4.28% |
| DPO | 19.78% | 24.90% | 18.18% | 5.12% | 2.86% | 5.23% | 3.07% |
| Final | 27.56% | 38.27% | 26.66% | 5.51% | 3.01% | 8.23% | 4.69% |

Base→SFT 的严格 UPOS、UAS、LAS 分别变化 +43.99、+4.23、+2.44 个百分点。SFT→DPO 的严格 UPOS 下降 25.52 点，但 UAS 上升 0.76 点；后者按句配对 bootstrap 的 95% CI 为 [+0.04, +1.47]，LAS 的 +0.39 点区间为 [−0.06, +0.83]。DPO→Final 的 UAS 与 LAS 差值区间均跨零。全部相邻比较使用 10,000 次按句配对重采样、seed 42；详细区间见 [`P0/dependency.md`](test/P0/dependency.md)。

P0 的 Task 1 完整表率依次为 2.12%、52.67%、19.74%、30.86%；预算截断率依次为 94.08%、0.14%、0%、0%。严格分数计入这些失败。完整 UPOS 报告见 [`P0/upos.md`](test/P0/upos.md)，逐句解析和格式诊断见各模型的 `sentences.jsonl`。

## 开发集选优与 P3 补充实验

从 dev 按 seed 42 无放回抽取 128 句；四模型共用 P0–P3，按四模型平均严格 Task 1 UPOS 选优（并列再看 LAS、候选顺序）。P0、P1、P2、P3 在 dev128 上的平均值分别为 **22.35%、36.31%、27.16%、56.17%**，故冻结 P3。P3 在 P2 规则之上添加一个固定 EWT train 示例，属于 **few-shot 补充实验**；示例句和完整选择记录见 [`data.json`](provenance/data.json)、[`selection.json`](provenance/selection.json)。选定后没有根据 test 调 prompt。

| 阶段 | P3 严格 Task 1 UPOS | 相对 P0 | P3 严格 UAS | 相对 P0 | P3 严格 LAS | 相对 P0 | P3 作者 UPOS/UAS/LAS F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Base | 0.00% | −1.31 点 | 0.00% | −0.12 点 | 0.00% | −0.03 点 | 15.44% / 9.20% / 5.98% |
| SFT | 72.61% | +27.31 点 | 35.79% | +31.43 点 | 24.09% | +21.61 点 | 76.44% / 38.89% / 25.65% |
| DPO | 68.94% | +49.17 点 | 35.06% | +29.95 点 | 23.89% | +21.02 点 | 62.03% / 33.88% / 22.83% |
| Final | 70.68% | +43.12 点 | 35.17% | +29.66 点 | 23.75% | +20.74 点 | 62.89% / 33.98% / 22.92% |

P3 提高了三个 Tülu checkpoint 的可解析词数和逐词分数，但没有形成 SFT→DPO→Final 的单调提升。P3 中 SFT→DPO 的严格 UPOS 为 −3.67 点（95% CI [−4.95, −2.44]）；UAS −0.73 点、LAS −0.20 点的区间均跨零。DPO→Final 的 UPOS 为 +1.74 点（[+0.95, +2.58]），UAS/LAS 区间跨零。详见 [`P3/upos.md`](test/P3/upos.md)和 [`P3/dependency.md`](test/P3/dependency.md)。

**Base P3 的严格零分应谨慎解释。** 其 2,077 句全部耗尽预算，输出不断重述表格、示例与 prompt；严格解析器对每句都记录 `ambiguous_task`，因此不擅自从重复表中挑选看似正确的预测。作者兼容评分会按不同恢复规则取到部分词，故得到非零 F1。两个口径衡量的可解析对象不同；不能把该 F1 当作严格三表成功率。

## 格式与树结构诊断

| 条件 | 阶段 | Task 1 完整表 | Task 3 完整表 | 截断句 | 有效树句 |
| --- | --- | ---: | ---: | ---: | ---: |
| P0 | Base / SFT / DPO / Final | 44 / 1,094 / 410 / 641 | 25 / 548 / 244 / 411 | 1,954 / 3 / 0 / 0 | 21 / 238 / 92 / 85 |
| P3 | Base / SFT / DPO / Final | 0 / 564 / 695 / 800 | 0 / 411 / 438 / 507 | 2,077 / 1 / 0 / 0 | 0 / 199 / 143 / 137 |

每项分母为 2,077 句。P3 下三个 Tülu 阶段的 Task 2→Task 3 HEAD 在双方均可比较的词上分别一致 **19,148/19,304、13,490/13,651、16,003/16,173**（约 99%）；这并不意味着树整体有效。P3 SFT/DPO/Final 分别检测到 819/898/912 个依存环、226/212/372 个多 root 句；树结构只作诊断，不改边，也不全句否决 UAS/LAS。作者口径对 P3 三个 Tülu 阶段分别因预测行数不符丢弃 406/719/687 句，另有 HEAD 越界与 root 标签修正；逐项计数在模型 JSON 的 `author.repair_audit`。

## 解释边界与复运行

这是一项沿用论文任务协议的**冻结推理行为评测**，不是论文任务微调成绩的复现。Base→SFT 同时改变权重与调用外壳，不能解释为纯 SFT 因果效应。三个 Tülu 阶段模板一致，但输出分数不证明内部句法表征变化。冻结权重不等于模型从未接触 EWT，不作“无污染”结论。P0 与含 train 示例的 P3 应分开陈述；P3 不冒充原论文协议。

复运行入口、环境、数据准备与评分命令见 [`README.md`](../README.md)。本仓库发布逐模型总分、标点辅助口径、作者恢复审计，以及逐句严格计数和错误诊断，见 [`results/test/`](test/)。逐句文件省略 gold 词数组和解析后的任务表；原始生成文本、输入及 token IDs 保存在原实验目录中，未包含于本次结果快照。[`optimized-comparison.json`](optimized-comparison.json)给出 P0→P3 的精确描述性差值。冻结哈希见 [`manifests/`](provenance/)。16 项本地测试通过；八个 test 单元的原始输出和逐句评分均已独立核对为 2,077 个唯一句子、25,094 个 gold 词且无基础设施失败，逐单元状态、停止原因与输出 SHA-256 见 [`final_audit.json`](provenance/final_audit.json)。

公开快照说明：以上实验数值保留原始结果。公开版移除了私人模型路径和环境锁文件中的本地构建路径；文件映射与 SHA-256 见 [`release-manifest.json`](release-manifest.json)。`frozen.json` 中的配置哈希属于原始运行配置，不能直接用于核验已改为相对路径的公开 `experiment.json`。
