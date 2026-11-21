# 目录与职责

先看入口和运行控制，再看规划、预处理/模型，最后看评估与审计。

```text
main.py                         统一入口
superagent/
  cli.py                        命令解析与分发
  config.py                     配置及资源边界
  runtime/
    orchestrator.py             实验循环、自动选型、最终重训
    worker.py                   隔离训练、多种子、失败恢复
    checkpoint.py               运行锁、恢复验证、预算与账本绑定
  planning/
    search_space.py             通用候选与模型构建
    llm.py                      上下文、API、提案解析与回退
    proposals.py                单类修改、参数边界和去重审核
    memory.py                   候选哈希与模型族统计
  preprocessing/
    datasets.py                 数据、标签、划分与输入哈希
    transformers.py             缺失值、编码、TF-IDF、SVD、图片表示
    feature_selection.py        方差、互信息及内存保护
  models/
    tabular.py                  提升树、融合、成员缓存
    neural.py                   神经网络、设备和重载
    text_encoder.py             固定 revision 的 MiniLM
  evaluation/
    metrics.py                  指标与方向
    inference.py                模型预测与最终评估
    nested_cv.py                训练内有限选参嵌套诊断
    split_overlap.py            训练/验证重叠描述
  governance/
    usage_ledger.py             token、费用、预留与预算
    frozen_bundle.py            冻结、哈希和 Unseen 时间窗
    test_access.py              配置测试文件的 Python 读取守卫
  common/json_io.py             原子 JSON 和带时间戳事件
scripts/
  workflows/                    批量任务、限次自动恢复
  data/                         NumPy 数据转换
  api/                          API 连通性检查
  validation/                   基准、产物和冻结流程检查
  reporting/                    真实事件生成决策时间线
tests/
  contracts/                    接口、课程和费用契约
  planning/                     搜索、提案与记忆
  preprocessing/                编码器与特征
  models/                       表格、文本和神经模型
  workflows/                    恢复、隔离、数据绑定与端到端行为
configs/                        示例、固定策略、模型和价格证据
docs/guides/                    操作与提交说明
docs/development/               建议复核、扩展和实验边界
docs/report/                    唯一报告 PDF 及 LaTeX
examples/                       小规模真实示例证据
```

## 一次运行

1. 入口读取配置，核对冻结策略、预算和时间窗。
2. 只读训练/验证，生成输入哈希、环境和重叠统计。
3. 规划器根据历史提出候选，审核边界和重复项。
4. 隔离 worker 拟合、验证，记录种子结果、模型族和费用。
5. 自动选择最终方案，合并训练/验证重训，之后允许测试预测。
6. 恢复沿父目录保留证据，验证版本、输入和原账本，继续剩余预算。

## 兼容和本地目录

根级 `superagent/features.py`、`tabular_models.py`、`torch_models.py`、`embeddings.py`、`selection.py` 只兼容历史 joblib 的类导入路径；新实现位于对应子目录，不在兼容层增加逻辑。

`scripts/` 根级旧脚本只转发旧命令，便于历史说明继续使用。

`data/`、`runs/`、`.model_usage/`、`.pretrained_cache/`、`releases/`、`submission_*/`、`pre/` 为本地输入、账本、缓存或产物，Git 忽略。课程数据目录保留原路径，避免破坏历史运行配置。
