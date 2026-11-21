# AIAA 3102 General Classification SuperAgent

同一程序根据训练数据、资源和验证反馈选择分类方法，支持表格、文本和图像。任务名称只组织输入与产物，不指定获胜模型。

## 从哪里开始

- [目录与职责](docs/architecture.md)：各层代码和数据流。
- [建议逐项复核](docs/development/recommendation_review.md)：采纳状态、证据与限制。
- [扩展功能及开发记录](docs/development/advanced_usage.md)：稳定性、特征筛选、嵌套诊断与恢复。
- [API 操作](docs/guides/api_usage.md)、[正式任务流程](docs/guides/full_assignment_workflow.md)。
- [英文报告和 LaTeX](docs/report/README.md)：报告对应的实验版本与构建方法。

## 安装与配置

Python 3.10+，在项目根目录的已激活环境中执行：

```bash
python -m pip install -r requirements.txt
python main.py doctor
```

GPU、XGBoost、MiniLM 等可选依赖见 `requirements-gpu.txt`。PyTorch 构建需匹配设备；`device: "auto"` 可回退 CPU。MiniLM 是默认关闭的冻结特征候选，不代替 Qwen/DeepSeek 规划语言模型。

CSV 训练数据包含标签列。配置相对路径以配置文件目录为准。模态、列名、ID 和指标按课程公告填写；参见 `configs/tabular.example.json`、`text.example.json`、`image.example.json`。`configs/final_policy.json` 用于冻结算法策略。

## 统一入口

```bash
python main.py train --config configs/tabular.example.json --output runs/task_new
python main.py predict --model runs/task_new/model.joblib --input data/test.csv --output runs/task_new/predictions_new.csv
python main.py evaluate --model runs/task_new/model.joblib --input data/test.csv --output runs/task_new/evaluation_new.json
```

输出目录/文件必须不存在。训练/验证用于搜索；最终方案确定后才进行测试预测。`evaluate` 需要标签，只用于最终评估。所有词表、编码、缩放和筛选只 fit 相应训练部分；最终可合并训练/验证重训。

## 批量任务和恢复

```bash
python scripts/workflows/run_released_batch.py --output runs/released_new --max-trials 10
python scripts/workflows/train_resilient.py --config configs/your_task.json --output runs/resilient_new --attempts 2
python main.py train --config configs/your_task.json --resume-from runs/interrupted --output runs/resumed_new
```

批量脚本读取本地 `in_vehicle_coupon_recommendation/` 与 `Released_Task2-Headlines_CTR/`。原始数据不上传 Git；测试数组只在选型重训后转换。上述批量命令默认本地规划，需要真实 LLM 时按 API 指南添加注册表、模型和本地凭据。

扩展参数包括 `--stability-repeats 2`、`--search-patience 6`、`--feature-search`、`--nested-cv-folds 2`、`--enable-minilm`。它们改变成本和搜索空间，不保证提高准确率。嵌套诊断在训练内部进行，不替换课程验证指标。

恢复绑定同一源码、配置、数据、依赖和原账本；停机时间计入原预算。仅恢复该版本产生的未完成运行，不能用新代码直接接续旧冻结版本。未知 API 消耗不会补零或盲目重放，详细限制见扩展指南。

## 产物与检查

`runs/<运行名>/` 保存模型、预测、trials/events/summary、环境、源码快照、实验记忆和费用快照。最终推理不需要规划 API 密钥；只加载可信来源的模型，并使用对应源码。

```bash
python -m unittest discover -s tests -v
python scripts/validation/check_frozen_workflow.py --help
```

源码按功能分层；少量根级旧模块只为历史模型兼容保留。旧脚本命令仍有转发入口，新文档使用分层路径。

## 结果和提交边界

原统一 Pro 开发快照：Coupon 79.07%，Headlines 66.86%。后续双种子及特征专项未超过原模型，本地提交包未替换。这些测试集曾被开发复用，不是独立无偏最终估计；性能分按组间排名计算。

历史超时用量与平台具体模型修订号仍需核实，报告保留 development draft 标识。报告、模型、源码和日志须来自同一版本；最新扩展不能套用旧报告成绩。AI 使用声明需成员确认。

Git 只上传报告 PDF `docs/report/report.pdf` 和 LaTeX 源码；其他 PDF、PPT、`pre/`、课程原始数据、运行产物、凭据及本地提交包均被忽略。
