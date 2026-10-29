# AIAA 3102 General Algorithm SuperAgent

根据 `AIAA3102 - Final Project - Full Assignment Requirements.md` 更新的通用分类 SuperAgent，支持表格、文本和图片分类、自主实验、可选 Qwen/DeepSeek 规划、费用记录和版本冻结。

**2026-10-23，当前 0.2.0 版本已通过 26 项单元测试，并在支持 CUDA 的开发环境上完成三种模态各 6 次候选实验。开发留出集准确率：表格 96.49%、合成文本 100%、手写数字图像 100%；均非课程正式成绩。学校 DeepSeek API 已通过独立连通性测试，正式 LLM 规划仍等待固定部署版本和历史官方人民币价格证据，不把未知费用记作零。**

正式运行流程、模型价格证据格式和冻结操作见 [`docs/full_assignment_workflow.md`](docs/full_assignment_workflow.md)。课程任务的数据格式、划分与指标仍需按每项任务的公告填写。

## 环境

要求 Python 3.10+。基础依赖见 `requirements.txt`，可选 PyTorch 依赖见 `requirements-gpu.txt`。可以使用独立虚拟环境或 Conda 环境，以下命令均在已激活的项目环境中执行。

CPU 环境可运行传统分类器；GPU 加速需要支持 CUDA 的 NVIDIA 显卡，以及与驱动兼容的 CUDA 版 PyTorch。程序自动检测 CUDA，并在支持时使用混合精度训练。具体依赖版本和设备信息由每次运行的环境日志记录。

先检查依赖导入情况，再按需安装。已有 CUDA 版 PyTorch 时，请保留与设备兼容的安装。

```powershell
python main.py doctor
# 如果实际导入检查发现缺少依赖，再决定是否执行：
# python -m pip install -r requirements.txt
```

## 输入和运行

每项任务使用一个 JSON 配置，见 `configs/`。数据统一为 CSV：表格数据每列一个特征；文本列存原文；图片列存图片路径。其他数据格式应先按课程规定转换成 CSV 清单，不能根据数据集名称定制模型。配置中的相对路径以配置文件所在目录为基准，CSV 中相对图片路径以 `image_root` 为基准。

```powershell
python main.py train --config configs/tabular.example.json --output runs/tabular_01
python main.py predict --model runs/tabular_01/model.joblib --input data/test.csv --output runs/tabular_01/predictions.csv
python main.py evaluate --model runs/tabular_01/model.joblib --input data/test.csv --output runs/tabular_01/final_evaluation.json
```

示例文件仅描述输入格式，没有附带课程数据。`--output` 训练目录必须不存在，避免覆盖实验。训练异常返回非零状态并保留已有日志。

配置字段：

- `data_type`：`auto` / `tabular` / `text` / `image`。auto 通过显式文本/图片列和列内容推断，不读取任务或数据集名称。含短文本时请明确指定 text。
- `train_csv`、`target_column` 必填；`validation_csv` 可选。不指定时从训练数据按类别分层留出 `validation_fraction`（默认 0.2）。每类至少需要两条样本，划分还需满足类别数约束。
- `test_csv` 可选，仅登记并检查文件路径不与训练/验证路径相同；训练过程中不打开它。测试集只能在确定最终方案后通过 `evaluate` 使用。
- `feature_columns` 可限制输入列；标签和 `id_column` 不能作为特征。可选 ID 列必须在每个输入文件中非空且唯一；若提供独立验证集，则检查两者 ID 不重叠。
- 文本用 `text_columns`；图片用 `image_column` 和 `image_root`。文本与图片模式只使用指定模态列，不融合其他列。
- `metric`：`accuracy`、`balanced_accuracy`、`f1_macro`、`f1_weighted`、`roc_auc`、`log_loss`。二分类 AUC 的正类默认为字符串标签按字典序最后一类，可通过 `positive_label` 显式设置。多分类 AUC 使用 macro OVR。
- `max_trials` 默认 16（包括基线）；`time_budget_seconds` 默认 1800；`trial_timeout_seconds` 默认 300；`final_fit_reserve_fraction` 默认 0.3；`threads` 默认 1；`seed` 默认 42。已有配置的显式预算始终优先。
- `enable_torch` 默认 true：PyTorch 可导入时增加表格 MLP / 图片 CNN 候选，缺少 PyTorch 时保留传统方法。`device` 为 `auto`（默认）、`cuda` 或 `cpu`，只影响神经网络候选；auto 在可用时选择 CUDA。
- `compute_profile` 默认 `auto`：CUDA 可用时启用 `gpu_large`，否则使用 `standard`。显式 `gpu_large` 要求 CUDA；`standard` 保留较小候选空间。高算力模式增加表格/文本残差 MLP、图片残差 CNN、更宽网络及三随机种子集成，并交替细调最佳 GPU 候选和总体最佳候选，避免一个小网络失利后不再探索 GPU。
- `neural_batch_size` 默认 256；`mixed_precision` 默认 true，在 CUDA 训练时优先使用 BF16，不支持 BF16 时使用 FP16 + GradScaler，CPU 使用 float32。预测使用 float32 张量，CUDA 库仍可能用 TF32 加速卷积，因此不承诺 CPU/GPU 概率逐位一致。核验脚本同时检查分类一致性、快速路径的概率误差和关闭 TF32 后的严格精度误差。
- `gpu_cache_fraction` 默认 0.2：预计输入大小低于当前空闲显存的这一比例且低于 2 GiB 时，将预处理后的训练数据一次性送入显存；否则逐批搬运。小型稀疏矩阵可在这个界限内转为稠密缓存。`cache_preprocessing` 默认 true，在单次运行内复用训练预处理结果；最终模型移除缓存路径依赖，预测不需要保留缓存。
- `evaluation_stage` 为 `development`（默认）、`released` 或 `unseen`。后两者要求冻结清单和共享 `evaluation_id`；Unseen 还要求带时区的官方发布时间。`evaluation_id` 只用于账本与日志，不参与候选选择。
- `llm_mode` 默认 `disabled`，保留无需 LLM 的自主规划；设为 `required` 时必须提供已核验的模型目录、固定模型 ID 和端点。模型仅允许符合日期要求的 Qwen/DeepSeek，单题预算 `llm_budget_rmb` 最大为 `20`；本地部署也按相同标准价格记账。密钥通过 `llm_api_key_env` 指定的环境变量读取。

## 自主流程与边界

1. 校验配置与训练/验证数据，记录内容哈希、特征统计、种子、源码版本哈希和环境。
2. 固定一份验证划分。所有预处理仅在各实验的训练部分拟合。
3. 先运行 DummyClassifier 基线，再比较该模态的通用方法。GPU 高算力模式使用更多模型容量和随机种子，同时保留传统模型作为竞争候选；按相同验证指标选择。
4. 高算力神经网络可训练 100–120 轮，使用 AdamW、梯度裁剪和余弦学习率；图片候选可加入轻微平移增强。早停监控集从当前实验训练部分再分层划出，外层验证集不参与梯度更新或早停；样本不足时禁用内部划分。预处理在整个外层训练部分拟合，因此内部早停监控不作为独立性能估计。
5. 后续根据外层验证得分调整学习率、轮数和正则化，记录候选理由、得分、失败和用时。失败或超时自动跳过，同一个候选不会重复尝试。最终模型按外层验证指标选择，使用训练和验证数据重新训练；神经网络使用训练内早停选出的轮数（集成则逐成员使用对应轮数），最终重训不再留出内部监控集。失败时自动尝试下一候选。
6. 最终模型生成后才允许独立预测/评估。评估命令不修改模型、不参与选择。

默认使用**基于规则和实验反馈的自主搜索**；新增可选 Qwen/DeepSeek 规划器，通过兼容 `chat/completions` 的远程 API 或本地服务，在程序生成的候选菜单中自动选择下一项实验。模型不会执行任意生成代码，规划上下文不含任务名称、数据集名称、列名、标签名称或测试成绩。`configs/model_registry.json` 暂为空，避免填入未经核实的模型身份、发布日期和价格；启用 LLM 前必须补齐官方证据并核验服务返回的模型身份和 token 用量。

CNN 从头训练，不下载预训练权重；大规模图像/语义任务可能仍需扩展方法。当前只支持独立同分布样本的分层划分；时间序列、分组主体或重复样本须提供合适的独立验证集。ID 重叠检查不能识别所有语义或近重复泄漏。

实验和最终训练分别在独立子进程中运行，可按墙钟超时终止。子进程从本次运行的 `source_snapshot/` 启动。父进程的数据读取、哈希、推理和落盘仅在阶段边界检查剩余时间，因此不是操作系统层面的硬时间隔离。线程数通过环境变量和模型参数设置；神经网络记录训练设备、损失和 PyTorch 峰值已分配显存，并将权重保存为 CPU NumPy 数组。CPU 时间、进程峰值内存、系统总显存尚未计量。

正式版新增共享 LLM 账本：按 2026-10-21 官方标准输入/输出价格和服务端实际 token 用量计费，忽略折扣；调用前按完整上下文规模预留最坏费用，推理输出同样纳入预留。预算不够则不发请求，转用规则规划；调用后缺少用量、超时或账本未核清时保留预留并停止，不把未知用量当作零。该保守预留可能拒绝实际很便宜的请求。目录和证据需要组员核验；软件校验来源域名、日期、快照哈希和金额结构，不会自动证明历史价格内容真实。

若训练配置显式指定 `device: "cuda"`，部署到 CPU 时需先在 Python 中加载模型并将 `model.named_steps["classifier"].set_params(device="cpu")`；使用 `device: "auto"` 的模型可自动选择设备。训练内早停、AMP 和固定随机种子不等于所有硬件和 CUDA 版本上位级一致。

输出包括：`model.joblib`、`manifest.json`、`profile.json`、`environment.json`、`source_snapshot/`、`events.jsonl`、`trials.jsonl`、`summary.json` 及每次实验的配置/结果/标准错误。启用缓存时另有 `preprocessing_cache/`，其中包含拟合数据的中间结果；可以在训练完成后自行清理，但请保留复现实验所需的原始数据和记录。manifest 记录数据和源码哈希，source_snapshot 保留同次运行的程序。只有最终模型成功保存后，summary 才记录 `completed`。

新增 `model_usage.json`（单任务所有 agent 的费用汇总）、`task_model_calls.json`（共享账本快照），启用 LLM 后还有 `model_calls.jsonl`、请求/响应文件及 `model_policy/` 的价格证据副本。正式 released/unseen 运行若配置了 `test_csv`，会在自动选型和重训完成后直接生成 `predictions.csv`，无需人工选择模型或另发预测指令。失败时保留候选日志；已完成但未完成输出的模型标为 `partial_model.joblib`，账目未核清的模型标为 `unverified_model.joblib`。

`predict` 输出 ID（若配置）和 `prediction`；支持概率的模型另输出 `probability_0` 等列，类别顺序见输出旁的 `.metadata.json`。`evaluate` 输出所选指标和 classification_report，并拒绝与训练/验证 CSV 内容哈希相同的文件。预测和评估输出必须是新文件。预测其他位置的图片清单时可以用 `--image-root` 重定位图片根目录。请从项目根目录运行，并保留 `superagent/`，因为序列化模型依赖其中的自定义预处理类。

## 提交材料

源码和统一入口已提供；真实数据、模型、结果、执行日志和 `report.pdf` 必须在正式任务发布后由完整自主运行产生。`docs/report_template.md` 是待填写的报告结构，不是实验结果；`AI_USAGE.md` 是需由组员核实的 AI 使用声明。提交前固定最终版本，用同一版本重新完成所有报告实验，保存数据划分、依赖和日志。

检查代码位于 `tests/`。旧版本曾通过 15 项检查；本次另加入 `tests/test_full_requirements.py`，覆盖费用、预算、模型资格、冻结策略、24 小时窗口和模拟 API 记账。**本次所有检查均未执行，不能据旧结果声称当前版本通过。** 后续允许运行时再执行 `python -m unittest discover -s tests -v`。

开发基准脚本是 `scripts/benchmark.py`，使用 scikit-learn 自带的乳腺癌表格数据、手写数字图片和合成多语言文本，无需下载。基线阶段只比较验证分数；优化阶段在自主选择完成后另外执行预测和独立留出集评估。数据位于 `data/development/`。`runs/development_baseline/` 保存初始 CPU 基线，`runs/development_optimized/` 保留包含图片预处理拟合状态错误的首轮诊断，`runs/development_final/` 保存修复后固定版本的三类任务模型、预测、评估和日志。训练数据、验证数据和开发留出集按 60% / 20% / 20% 分层划分；它们只是开发用途，不能用于声称课程正式任务性能。

上一轮轻量版本的 24 次候选实验全部成功，三项任务使用相同源码哈希。开发留出集准确率：表格 96.49%（114 条）、合成文本 100%（180 条）、手写数字图片 97.78%（360 张）。对应验证准确率分别是 99.12%、100%、98.89%。当时 MLP 和 CNN 的验证准确率低于最佳传统方法，因此保留了逻辑回归和 ExtraTrees。这些旧结果和旧源码快照保留在 `runs/development_final/`。

`runs/development_final/verification.json` 记录了模型重新加载后预测一致、概率归一化、源码快照一致、训练数据评估拦截、输出覆盖拦截和超时子进程终止检查。此次实际修复了 scikit-learn 1.9 下图片预处理器缺少拟合标记导致 CNN 流水线预测失败的问题；另移除了 CNN 非确定性池化操作及文本向量器的无效参数警告。

2026-10-18 高算力开发版（旧源码）的产物位于 `runs/development_gpu_large/`：当时 15 项检查通过，三个任务共 36 次候选实验全部成功，其中 19 次为 CUDA 神经网络候选（集成候选另包含多个成员）。三个任务使用同一旧源码哈希。图片任务自动选中宽度 48、分类隐藏层 384 的残差 CNN，验证准确率从轻量版本的 98.89% 提升到 99.72%，360 张开发留出图片准确率从 97.78% 提升到 100%。表格开发留出准确率为 96.49%，合成文本为 100%，两者选择传统模型。这些旧结果不是当前 0.2.0 版本的运行结果。

高算力版本的 `verification.json` 同时确认最终图片模型已使用训练和验证数据全量重训、CPU/GPU 分类结果一致，以及保存/加载、文件覆盖拦截和超时终止正常。默认 TF32 GPU 卷积相对 CPU 的最大概率差为约 0.000523；关闭 TF32 的精度对照差为约 0.00000143。`unit_tests.log` 保存该版本 15 项检查的实际输出。

下方为运行命令示例；修复自动预测完成事件的日志字段冲突后，同一最终源码的新开发结果在 `runs/development_full_requirements_final/`，冻结流程验证在 `runs/frozen_workflow_final_20261023/`。此前 smoke 目录保留开发排错记录，不能混用源码哈希。旧运行的源码与当前版本不同，当前源码一致性检查会拒绝将旧结果当作新版本结果。

```powershell
# 输出目录必须不存在；复验时使用新的 --name。
python scripts/benchmark.py --phase baseline --name repeat_cpu
python scripts/benchmark.py --phase optimized --name repeat_cuda
# GPU 较高算力配置示例；每类 12 个候选，单任务最多 20 分钟：
python scripts/benchmark.py --phase optimized --compute-profile gpu_large --max-trials 12 --budget 1200 --name repeat_gpu_large
# 核验本版本新生成的运行（timeout_check 目录必须不存在）：
python scripts/check_artifacts.py --run repeat_gpu_large
```
