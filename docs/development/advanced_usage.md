# AIAA 3102 General Algorithm SuperAgent

面向表格、文本和图像分类的通用自主实验系统。输入数据和评价指标后，程序自动分析数据、比较候选模型、根据验证结果调整方案，并保存最终模型与预测结果。

表格搜索比较 CatBoost、LightGBM、传统树模型、线性模型和可用的神经网络，并探索多随机种子集成及不同提升树的概率融合。类别值和缺失值在训练流程中统一处理；类别统计与编码只在拟合数据中建立，不使用外部验证或测试数据。候选容量、迭代数和正则化根据验证结果自主调整，不根据数据集名称选择算法。

多个文本列还会比较保留列位置的向量拼接；恰有两列时，增加共享词表的有符号差分，避免拼接词袋丢失列间关系。GPU 环境可额外比较 XGBoost 和 CPU CatBoost／GPU XGBoost 融合；使用 `requirements-gpu.txt` 安装可选依赖。所有新候选仍按同一验证指标自动比较。

文本搜索也包含词级与字符级联合表示；融合模型可比较不同成员权重，并缓存相同训练数据与参数下的成员，避免重复训练。最终模型不依赖训练缓存。

## 1. 安装环境

要求 Python 3.10+。在项目根目录、已激活的 Python 环境中执行：

```bash
python -m pip install -r requirements.txt
python main.py doctor
```

如需 GPU 训练，安装与设备和驱动兼容的 CUDA 版 PyTorch。可选依赖见 `requirements-gpu.txt`。`device: "auto"` 会自动选择可用设备；没有 GPU 时可使用 CPU。

## 2. 准备数据和配置

输入使用 CSV，配置中的相对路径以配置文件所在目录为基准。

- 表格：每行一个样本，每列一个特征。
- 文本：通过 `text_columns` 指定文本列。
- 图像：通过 `image_column` 指定图片路径列，通过 `image_root` 指定图片目录。
- 训练数据包含标签列；预测数据不要求包含标签。可选 ID 列必须非空且唯一。

参考 `configs/tabular.example.json`、`configs/text.example.json` 和 `configs/image.example.json`。以表格为例：

```json
{
  "data_type": "tabular",
  "train_csv": "../data/train.csv",
  "validation_csv": "../data/validation.csv",
  "test_csv": "../data/test.csv",
  "target_column": "label",
  "id_column": "id",
  "metric": "f1_macro",
  "device": "auto",
  "max_trials": 16,
  "time_budget_seconds": 1800
}
```

没有独立验证集时可省略 `validation_csv`，程序从训练集分层划分。支持 `accuracy`、`balanced_accuracy`、`f1_macro`、`f1_weighted`、`roc_auc` 和 `log_loss`；按每项课程任务的公告选择指标。

## 3. 训练、预测和评估

```bash
python main.py train --config configs/tabular.example.json --output runs/task_01
python main.py predict --model runs/task_01/model.joblib --input data/test.csv --output runs/task_01/predictions.csv
python main.py evaluate --model runs/task_01/model.joblib --input data/test.csv --output runs/task_01/evaluation.json
```

输出目录和输出文件必须不存在。`predict` 生成 ID、预测类别和类别概率；`evaluate` 需要标签列，生成最终得分与分类报告。

程序自动执行候选比较和最终选型，不需要在运行中手工调参或挑选模型。测试集不参与训练、调参或方案选择，只在最终方案确定后使用。正式 `released` / `unseen` 运行如果配置了 `test_csv`，训练结束会自动生成预测文件，无需再次执行 `predict`。

## 4. 输出文件

每次运行保存最终模型、数据分析、候选实验、选择理由、环境信息与源码快照，主要包括：

- `model.joblib`：包含预处理步骤的最终模型。
- `predictions.csv`：预测结果，类别顺序见旁边的 metadata 文件。
- `trials.jsonl`、`events.jsonl`：实验结果与执行过程。
- `summary.json`：完成状态、最终方案及验证得分。
- `manifest.json`、`environment.json`、`source_snapshot/`：复现所需信息。

失败时保留诊断日志；部分结果会明确标记。加载模型时需保留项目源码，且只加载可信来源的 joblib 文件。

## 5. 使用 API

默认无需 API，程序使用规则与验证反馈规划实验。需要接入语言模型时，参见 [API 使用方法](../guides/api_usage.md)，包括学校接口、密钥读取和连通性测试。

## 6. 课程任务与提交

同一 SuperAgent 处理所有任务，不通过任务名或数据集名分发到人工定制方案。报告成绩应来自完整自主运行，运行中不得人工改代码、安排下一项实验或挑选最终模型。

提交前固定最终源码和算法策略，使用同一版本完成报告实验。Unseen Task 在官方发布后的 24 小时内使用冻结版本运行。操作步骤见[正式任务运行流程](../guides/full_assignment_workflow.md)。

提交材料包括源码与统一入口、依赖与 README、各任务模型和结果、执行日志、`report.pdf` 及 AI 工具使用声明。报告结构见 `docs/guides/report_template.md`；原始课程要求见 `AIAA3102 - Final Project - Full Assignment Requirements.md`。

## 7. 检查

已提供任务数据时，可自动使用同一冻结版本运行两项 Released Tasks：

```bash
python scripts/workflows/run_released_batch.py --output runs/unified_run --max-trials 10
```

脚本读取项目根目录下的 `in_vehicle_coupon_recommendation/` 和 `Released_Task2-Headlines_CTR/`，依次完成训练、预测与最终评估。NumPy 数据转换保持原始划分和行顺序；测试数据只在该任务选型、重训后读取。数据和运行产物不包含在 Git 仓库中，输出目录必须不存在。

```bash
python -m unittest discover -s tests -v
```

本地开发基准可通过 `scripts/validation/benchmark.py` 运行；使用内置数据和合成文本，不代表课程正式成绩。GPU 网络从头训练，不下载预训练权重。

可选安装 `requirements-gpu.txt` 中的 `sentence-transformers`，并在冻结策略中显式设置 `enable_minilm: true`（批量脚本使用 `--enable-minilm`）后，程序会发现通用 MiniLM 冻结特征候选，在表格和文本分类中按验证结果与现有方法比较。它不是规划语言模型，不使用任务名称分发；固定仓库 revision 见 `superagent/models/text_encoder.py` 和候选日志。首次使用自动联网下载，需可写的 Hugging Face 缓存；离线复现应保留缓存。表格使用字段和值的文本表示，文本保留列顺序；分类器只在允许的拟合数据上训练。不保证该候选能提高分数，最终模型仍由验证结果决定。

## 8. 本项目如何对应课程评分标准

- General Design（30%）：`superagent/planning/llm.py` 基于已完成实验的验证反馈，要求输出 `diagnosis`、`next_action` 和放弃策略；连续两次失败的模型族退出候选菜单。记录在 `events.jsonl`，不声称模型能执行任意生成代码。
- Released / Unseen Performance（20% / 10%）：`agent.py` 自主验证选型，`prediction.py` 只做推理与最终评估。`partial/vNNN/` 保存逐次改进的部分模型及预测；后续失败时保留已有版本。批量脚本移除了历史硬编码成绩。
- Execution / Reproducibility（10%）：冻结源码、输入哈希、调用账本、`summary.json` 和 `tests/workflows/test_resilience.py`。设备自动选择；分配失败时减小 batch，再尝试 CPU；无法恢复的异常保留诊断。
- Presentation（30%）：运行 `python scripts/reporting/decision_timeline.py <events.jsonl路径> --output Timeline.md` 提取真实决策时间线，不补写不存在的反思，不擅自改变课程指定指标。

默认最多 10 次实验，同时受时间和费用限制；`max_cost_rmb` 默认 15 元，下一次最坏情况调用无法容纳时停止搜索，任务累计总额仍不超过 20 元。正式策略 `configs/final_policy.json` 启用 Flash API，密钥按 API 使用说明通过环境变量提供；开发测试可显式关闭 API。

`test_set_used_for_selection: false` 表示测试标签与成绩不进入搜索。开启 `progressive_output` 时，部分模型固定后可提前预测测试特征；日志明确标记 `partial_test_read_started`。最终模型的 `selection_completed` 先于最终 `test_read_started`，单元测试断言该顺序。若要求整个搜索结束前完全不读取测试文件，请设置 `progressive_output: false`，仍可保留最终产出流程。

## 9. 无 GPU 端到端演示

`examples/cpu_smoke/` 使用 scikit-learn 自带 Iris 数据的固定分层划分，提供真实预测、摘要和事件日志；它只是运行证据，不能替代课程任务成绩，也不证明真实 LLM 调用。该演示显式禁用 CUDA 和 LLM，调用数为零。

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
python main.py train --config examples/cpu_smoke/task.json --output runs/cpu_smoke_new
python main.py predict --model runs/cpu_smoke_new/model.joblib --input examples/cpu_smoke/test.csv --output runs/cpu_smoke_new/reloaded_predictions.csv
```

模型保存在上述本地输出目录；仓库保存脱敏后的真实日志、预测和复现输入，完整源码快照及环境信息保留在原运行目录。

## 正式提交的测试隔离与轮数比较

新增受约束实验提议：LLM 可选一个通用候选作为种子，再通过 `proposal` 提出允许范围内的参数变体，同时给出 `hypothesis` 和 `success_criterion`。`superagent/planning/proposals.py` 拒绝修改模型族、代码、路径、指标、资源及范围外参数；提议仍由验证反馈检验，而不是预先宣称有效。没有使用任务名或数据集身份定制搜索。

多列文本增加通用概率融合候选，复用有序字符逻辑回归与词/字符校准 SVM。新增能力、参数变体及融合均属于搜索空间；不能据此直接断言准确率提高。Pro 规划输出上限默认提高到 4096 tokens，可通过 `--llm-output-tokens` 设置，实际 token 和费用仍逐次记录。

正式策略和批量脚本默认 `progressive_output: false`：搜索期间不读测试文件、不生成中间测试预测，最终选型完成后才推理与评估。开发者如显式开启中间产出，须单独标注该运行，不能冒充严格隔离的正式结果。

`feedback_assessed` 将每轮验证结果与此前最优值对照，记录真实增益或停滞；结合 `llm_plan_selected` 的诊断与下一步理由形成证据链。不能事后编写不存在的 LLM 推理，也不保证多跑必然提高测试分数。

比较 5、10、15 轮时，优先使用同一次 15 轮运行的验证集最佳值前缀，避免重复读取测试集和重复调用。最佳验证值随轮数不下降是选型规则的结果，不等于泛化性能必然改善。测试成绩只在最终选型后评估，不能用来选择轮数。

2026-11-19 开发实验：每项 10 次搜索，MiniLM 未超过原有最佳验证结果；新运行测试准确率 Coupon 79.07%、Headlines 66.06%，未替换已有提交模型（79.07%、66.86%）。固定轮数下新增候选占用搜索预算，不能将这次对比当作匹配预算的因果消融。MiniLM 默认关闭。

## 反馈搜索增强（2026-11-20）

- `trial_memory.json` 保存本轮候选的规范 JSON 哈希和模型族状态；同一模型族与参数组合不再提交。`proposals.py` 在执行前再次检查重复与参数边界。
- 一次 LLM 覆盖提议只允许修改特征设置或超参数中的一类；模型切换通过候选菜单进行。上下文保留最近五轮、最优方案、模型族摘要及上次拒绝原因，并提供两个 JSON 示例。上下文过大时先缩小本轮明确列出的可执行菜单。
- 模型族摘要记录候选间最佳/均值/标准差、失败次数与耗时。两次失败或至少三次候选表现明显落后时退役；六轮后倾向分配给表现较好的族，每三轮保留一次原顺序探索。这是启发式资源分配，不是统计显著性检验。
- `stability_repeats` 可设置 1–3（默认 1），每个候选用不同训练种子在同一指定验证集上测量，`score` 为均值，另记 `validation_std` 和每个种子成绩。没有更换官方验证划分；种子标准差不是泛化误差或置信区间。最终模型按验证均值选型，使用基础种子重训，不自动集成这些重复。
- `search_patience` 默认为 0（关闭）；启用后至少完成八轮，连续指定轮数没有提高验证均值时提前停止，为最终重训保留时间。预算/时间闸门仍然有效。
- 当配置含 `test_csv` 且关闭中间预测时，Python audit hook 在当前解释器中阻止该路径在最终选型前被打开。它不是跨进程安全沙箱，不承诺阻止本机恶意原生代码或文件别名。批量转换器仍在训练完成后才转换/读取测试数据。
- `planner_checkpoint.json` 保存结构化规划记忆与拒绝状态；`trial_memory.json` 保存全部已完成候选状态。原始请求与回复继续保存在 API 日志中。该阶段尚未提供恢复入口；后续扩展已经加入，见下节。

开发比较示例（历史费用未知时仍是 provisional 开发结果）：

```bash
python scripts/workflows/run_released_batch.py --output runs/stability_new --max-trials 10 --stability-repeats 2 --search-patience 6 --trial-timeout 1200 --llm-registry configs/model_registry.pro.json --llm-model DeepSeek-V4-Pro --key-file <本地密钥文件> --provisional-accounting --llm-output-tokens 2048
```

前一轮 `memory_stability_20261120` 未启用后续新增的嵌套诊断和互信息筛选。当前仍不按臆测的新闻时间/组别重切分，也不自动安装依赖。不会从标题内容臆造时间或用测试成绩调搜索策略。没有对未知消耗的 API 请求自动重试，避免重复收费和破坏任务总预算。已有提交包的模型与冻结代码不会被开发实验覆盖。

本轮实测见 `runs/memory_stability_20261120/comparison.json`：CPU、双种子验证，Coupon 完成 10 轮、Headlines 在 9 轮因连续六轮均值未提升而停止。最终测试准确率分别为 78.99%、66.06%，没有替换原有提交模型。API DNS 连接失败，实际采用本地规划，不能据此声称新 prompt 已提高 LLM 决策质量。当前 48 项检查中 46 项通过，2 项 CUDA 检查跳过。

## 恢复、特征组件与嵌套诊断扩展

### 无人值守恢复

```bash
python scripts/workflows/train_resilient.py --config configs/your_task.json --output runs/resilient_new --attempts 2
python main.py train --config configs/your_task.json --resume-from runs/interrupted --output runs/resumed_new
```

监督入口最多尝试三次，每次写入新目录。显式 Ctrl+C 不会被自动重启；异常退出后自动恢复。恢复必须使用原配置与相同源码、Python/依赖、训练/验证内容和任务账本。停机与等待遗留 worker 的时间计入原预算，Unseen 的官方截止时间不会重置。运行锁阻止并发恢复；不能确认 worker 状态或预算已用尽时安全拒绝。

已提交完成的运行不能恢复。中断前已完成的 trial 从日志恢复；已启动但未得到结果的方案记为 interrupted，保留在去重记忆中；遗留 worker 已写出的结果会恢复。恢复会重建模型族状态、保留最近结构化诊断与拒绝原因，原 API 请求/回复通过父目录链保留。恢复期间不得删除父目录。它不会重放用量未知的 API 请求，不会重置费用账本；重新拟合最终模型仍可能需要计算时间。

### 可选特征与训练内部诊断

`enable_feature_search: true` 启用额外通用线性候选：无筛选、方差筛选、互信息筛选。互信息稠密化受 128 MiB 上限保护；多列文本候选先做训练内拟合的 64 维 SVD，再筛选。超出范围会记录失败并继续其他候选。

LLM 可在允许范围内提议 `imputer`、`encoding`、`feature_filter`、`percentile`、`vectorizer_options`；后者支持 n-gram 范围、min_df、max_features、是否使用 IDF、binary 和可选英文停用词。没有把缺失值或词表统计拟合到测试集。文本与表格专属设置不能交叉使用，仍遵守一次只改变特征或超参数一类的约束。

`nested_cv_folds: 2` 或 `3` 启用**训练集内部有限正则化选择的嵌套诊断**：默认最多分层抽取 600 行，内层两折在两个 C 值中选择，外层独立评估。仅适用于受支持的逻辑回归/线性 SVM，稀有类别不足时明确跳过。它不是完整 AutoML 搜索的嵌套评估，也不替代课程指定验证集或最终测试指标。抽样诊断分数不能直接与全量拟合成绩比较。可用 `nested_cv_max_rows` 调整样本上限。

`split_risk.json` 描述训练/验证的完全相同特征行、共享文本值数量，不输出文本内容，不读取测试。共享新闻标题可以是任务正常结构；这个统计不能证明存在泄漏，也不能推断新闻时间。

专项比较命令：

```bash
python scripts/workflows/run_released_batch.py --output runs/features_new --max-trials 4 --feature-search --nested-cv-folds 2 --stability-repeats 2 --trial-timeout 1200
```

API 用法和费用约束仍见 `docs/guides/api_usage.md`。本次 `runs/feature_nested_20261120` 是四轮特征专项开发试验，不是覆盖原十轮搜索空间的正式运行。原有提交包模型不会因此替换。

恢复只支持同一源码版本产生的中断运行，不能直接接续旧版本或已完成的实验。恢复后的第一个决策使用本地规划，避免重放可能已收费的中断 API 请求。少量种子的标准差只作为诊断，不机械加入任意惩罚项；当前没有可靠时间/来源组别元数据，因此保留官方划分，并提供训练内诊断与重复值统计。

最新验证：53 项检查中 51 项通过、2 项 CUDA 检查跳过。`runs/resume_fault_probe_20261120_v2/verification.json` 记录真实进程中断后自动恢复完成且原账本路径保持不变。`runs/feature_nested_20261120/comparison.json` 记录四轮线性特征专项比较：Coupon 66.93%、Headlines 65.75%，未超过原提交模型。Headlines 训练/验证间未发现标准化后的重复标题或完全相同特征行；这不排除语义近重复等其他风险。主提交包模型保持原样。

新增运行功能使用既有 scikit-learn/SciPy/pandas 和 Python 标准库，不新增 Python 依赖。故障注入测试另外使用 Windows PowerShell Get-CimInstance 查询测试子进程，taskkill 仅终止该测试创建的训练进程树；普通训练与恢复无需这些故障注入工具。
