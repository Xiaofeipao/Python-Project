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

默认无需 API，程序使用规则与验证反馈规划实验。需要接入语言模型时，参见 [API 使用方法](docs/api_usage.md)，包括学校接口、密钥读取和连通性测试。

## 6. 课程任务与提交

同一 SuperAgent 处理所有任务，不通过任务名或数据集名分发到人工定制方案。报告成绩应来自完整自主运行，运行中不得人工改代码、安排下一项实验或挑选最终模型。

提交前固定最终源码和算法策略，使用同一版本完成报告实验。Unseen Task 在官方发布后的 24 小时内使用冻结版本运行。操作步骤见[正式任务运行流程](docs/full_assignment_workflow.md)。

提交材料包括源码与统一入口、依赖与 README、各任务模型和结果、执行日志、`report.pdf` 及 AI 工具使用声明。报告结构见 `docs/report_template.md`；原始课程要求见 `AIAA3102 - Final Project - Full Assignment Requirements.md`。

## 7. 检查

已提供任务数据时，可自动使用同一冻结版本运行两项 Released Tasks：

```bash
python scripts/run_released_batch.py --output runs/unified_run --max-trials 10
```

脚本读取项目根目录下的 `in_vehicle_coupon_recommendation/` 和 `Released_Task2-Headlines_CTR/`，依次完成训练、预测与最终评估。NumPy 数据转换保持原始划分和行顺序；测试数据只在该任务选型、重训后读取。数据和运行产物不包含在 Git 仓库中，输出目录必须不存在。

```bash
python -m unittest discover -s tests -v
```

本地开发基准可通过 `scripts/benchmark.py` 运行；使用内置数据和合成文本，不代表课程正式成绩。GPU 网络从头训练，不下载预训练权重。

## 8. 本项目如何对应课程评分标准

- General Design（30%）：`superagent/llm_planner.py` 基于已完成实验的验证反馈，要求输出 `diagnosis`、`next_action` 和放弃策略；连续两次失败的模型族退出候选菜单。记录在 `events.jsonl`，不声称模型能执行任意生成代码。
- Released / Unseen Performance（20% / 10%）：`agent.py` 自主验证选型，`prediction.py` 只做推理与最终评估。`partial/vNNN/` 保存逐次改进的部分模型及预测；后续失败时保留已有版本。批量脚本移除了历史硬编码成绩。
- Execution / Reproducibility（10%）：冻结源码、输入哈希、调用账本、`summary.json` 和 `tests/test_resilience.py`。设备自动选择；分配失败时减小 batch，再尝试 CPU；无法恢复的异常保留诊断。
- Presentation（30%）：运行 `python scripts/decision_timeline.py <events.jsonl路径> --output Timeline.md` 提取真实决策时间线，不补写不存在的反思，不擅自改变课程指定指标。

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
