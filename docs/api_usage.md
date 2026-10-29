# API 使用方法

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

`configs/final_policy.llm.example.json` 提供接入参数示例。训练任务配置使用 `llm_mode: "required"`，并填写 `llm_model_id`、`llm_base_url`、`llm_api_key_env` 和 `llm_registry`（已有模型配置文件路径）。该示例中的模型配置目录目前为空，需完成模型配置后才能用于规划；连通性测试不依赖该目录。

完整规划器从 `HKUST_API_KEY` 环境变量读取密钥，因此训练进程应在已设置此变量的会话中启动。规划器接收数据统计和验证反馈，从通用候选菜单中选择下一项实验；不会向接口发送测试集或执行任意生成代码。

使用结束后可清除当前会话凭据：

```powershell
Remove-Item Env:HKUST_API_KEY
```
