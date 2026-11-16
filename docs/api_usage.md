# API 使用方法

## Pro 开发实验

`configs/model_registry.pro.json` 使用 DeepSeek-V4-Pro，按官方 2026-08-17 生效的空闲价格核算：输入（缓存未命中）4.5 元/百万 token、输出 13.5 元/百万 token。价格原图和官方输出上限说明随注册表保存并校验哈希；学校接口返回的具体部署修订号仍未独立确认。

```powershell
python scripts/run_released_batch.py --output runs/pro_experiment --max-trials 10 --llm-registry configs/model_registry.pro.json --llm-model DeepSeek-V4-Pro --key-file $credentialPath --provisional-accounting
```

`--provisional-accounting` 仅允许开发阶段使用：历史超时仍保持未知，完整预留其最坏费用；预算按“已知费用 + 未知预留 + 下一次最坏预留”控制，不另开任务账本、不把未知值置零。模型的官方输出上限用于保守预留，实际费用仍只按真实 token 计算。若再发生网络超时，不重试该请求，后续可在预算允许时继续或退回本地规划。

这类实验的 `summary.json` 会标记 `accounting_provisional: true`、`eligible_for_formal_submission: false`，累计实际费用保持 `null`。结果只能作为开发实验；正式 released/unseen 模式禁止该选项，仍要求核清每次实际用量。测试文件仍只在最终选型后读取。

## 学校接口

- Base URL：`https://aigc-api.hkust-gz.edu.cn/api/v2`
- 请求模型：`DeepSeek-V4-Pro`
- 认证：`Authorization: Bearer <密钥>`

程序自动在 Base URL 后追加 `/chat/completions`，不要重复填写该后缀。远程连接使用 HTTPS。

## 读取密钥并测试

在项目根目录、已激活的 Python 环境中运行以下 PowerShell 命令。按提示输入现有密钥文件路径，不要把密钥文件放入 Git 仓库。

```powershell
$credentialPath = Read-Host '本地 API 密钥文件路径'
$env:HKUST_API_KEY = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $credentialPath).Path).Trim()
python scripts/probe_hkust_api.py --output runs/api_probe_01
```

输出目录必须为新目录。成功时返回 `status: "success"`，请求与响应摘要保存在该目录。也可直接传入密钥文件：

```powershell
python scripts/probe_hkust_api.py --key-file $credentialPath --output runs/api_probe_02
```

如果代理影响学校内网访问，可仅为当前进程配置直连域名：

```powershell
$env:NO_PROXY = (@($env:NO_PROXY, 'aigc-api.hkust-gz.edu.cn') | Where-Object { $_ }) -join ','
```

## 用于实验规划

已提供 Flash 配置 `configs/model_registry.flash.json`。它采用官方 2026-09-10 生效的空闲时段价格：每百万 token 输入 1 元、输出 4 元；全部输入按缓存未命中计算。此为指定核算口径，不代表学校平台实际扣费。官方原图及 SHA-256 随配置保存。平台返回 `DeepSeek-V4.1-Flash`，具体修订号尚未独立确认。

使用同一冻结版本运行两个任务并启用 API 规划：

```powershell
$credentialPath = Read-Host '本地 API 密钥文件路径'
python scripts/run_released_batch.py --output runs/flash_run --max-trials 24 --llm-registry configs/model_registry.flash.json --key-file $credentialPath
```

每个任务共用持久化账本，限额 20 元。输出中的 `model_usage.json`、`task_model_calls.json` 和 `model_calls/` 记录实际 token（含返回的推理用量）、核算费用、请求与响应。密钥仅在进程环境中使用，不复制到输出。API 只负责实验规划，本地训练仍可使用 GPU。

`configs/final_policy.llm.example.json` 提供接入参数示例。训练任务配置使用 `llm_mode: "required"`，并填写 `llm_model_id`、`llm_base_url`、`llm_api_key_env` 和 `llm_registry`（已有模型配置文件路径）。默认注册表 `configs/model_registry.json` 已包含 Flash 的官方证据与固定空闲费率；学校部署具体修订号仍待核实。

完整规划器从 `HKUST_API_KEY` 环境变量读取密钥，因此训练进程应在已设置此变量的会话中启动。规划器接收数据统计和验证反馈，从通用候选菜单中选择下一项实验；不会向接口发送测试集或执行任意生成代码。

使用结束后可清除当前会话凭据：

```powershell
Remove-Item Env:HKUST_API_KEY
```

每次 API 尝试会刷新任务累计 `usage.jsonl`；`summary.json` 的 `model_usage` 汇总 token、已知费用和未知用量预留。历史失败不会删除，未知费用不会当作零。
