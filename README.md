# AIAA 3102 General Algorithm SuperAgent

面向表格、文本和图像分类的通用自主实验系统。输入数据和评价指标后，程序自动分析数据、比较候选模型、根据验证结果调整方案，并保存最终模型与预测结果。

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

```bash
python -m unittest discover -s tests -v
```

本地开发基准可通过 `scripts/benchmark.py` 运行；使用内置数据和合成文本，不代表课程正式成绩。GPU 网络从头训练，不下载预训练权重。
