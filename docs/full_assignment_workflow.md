# 正式任务运行流程

## 1. 准备任务输入

按课程公告准备训练、验证和测试数据，填写标签列、模态列、指标和任务 ID。使用相同 SuperAgent 处理各任务，不能按任务或数据集名称选择人工定制方案。测试集只用于最终预测或评估。

## 2. 固定提交版本

开发结束后，将算法设置写入 `configs/final_policy.json`，创建冻结副本：

```bash
python main.py freeze --policy configs/final_policy.json --output releases/final
python releases/final/main.py verify-freeze --manifest releases/final/frozen_manifest.json
```

输出目录必须不存在。冻结包包含程序、算法策略与文件哈希。后续正式运行使用冻结入口，不能修改包内代码、策略或示例文件。任务输入配置放在冻结包之外。

## 3. 运行 Released Tasks

参照 `configs/released.example.json`，填写真实数据路径、指标、唯一 `evaluation_id` 和冻结清单路径。正式入口需要持久共享运行目录，在 PowerShell 中设置：

```powershell
$env:SUPERAGENT_USAGE_ROOT = Join-Path (Get-Location) '.model_usage'
python releases/final/main.py train --config configs/released.example.json --output runs/released_task_01
```

示例配置包含占位信息，填写后再运行。程序自动比较候选、重训最终模型；配置了 `test_csv` 时自动保存预测。需要最终得分且测试标签已公开时，可单独运行 `evaluate`，结果不能反馈给搜索。

## 4. 运行 Unseen Task

两组课程展示结束并收到正式任务后，参照 `configs/unseen.example.json` 填写数据、指标和带时区的官方发布时间 `unseen_released_at`：

```bash
python releases/final/main.py train --config configs/unseen.example.json --output runs/unseen_task_01
```

从官方发布时刻起计算 24 小时，不从程序启动时重新计时。使用提交时冻结的版本，不得在运行中人工改代码、尝试顺序或最终模型；在截止前提交预测和执行日志，无需另交报告。

## 5. 整理提交材料

- 完整源码、统一入口、依赖文件及 README。
- Released Tasks 的最终模型、预测或评价结果。
- 执行日志、候选实验记录、环境信息和源码快照。
- 项目报告 `report.pdf` 和 AI 工具使用声明。

所有报告结果必须来自同一最终提交版本的完整自主运行；失败任务说明阶段、原因及有效的部分结果。具体提交截止为所在 section 最后一周第一次课前一晚 23:59，日期按课表确认。

API 接入操作单独见 [API 使用方法](api_usage.md)。原始课程要求保留在项目根目录，提交时按课程公告核对。
