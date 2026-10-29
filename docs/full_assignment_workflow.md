# 正式要求对应的运行流程

依据：`AIAA3102 - Final Project - Full Assignment Requirements.md`。

**状态：2026-10-23 后续获得运行授权，当前版本已通过 26 项单元测试与三模态 GPU 开发试跑。生产模型目录仍为空；启用 LLM 需填写模型发布日期与标准单价。本文正式提交路径是操作示例，不是已经提交给课程的产物。**

## 1. 本次落实的规则

- 模型调用只接受核验目录中的 Qwen / DeepSeek；发布日期不得晚于 **2026-10-21**，必须有官方标准 API 价格。这里的 token 计价模块用于语言模型；现有传统分类器和从头训练的神经网络没有新增外部模型调用。
- 每个评估任务最多 **RMB 20**。所有调用按截止日的官方标准输入、输出价和实际 token 计费，不考虑缓存折扣、赠送额度或供应商促销；本地服务也一样。
- 同一任务所有 agent 和多次启动共用账本。不能换输出目录或冻结副本来重置额度。
- Released Tasks 的最终报告结果及 Unseen Task 必须来自同一冻结程序与算法策略。任务配置只填写数据、格式、目标、指标、共享任务 ID 及运行阶段等输入信息。
- 指定测试集只在最终选型后用于预测或最终评估。正式运行设置 `test_csv` 后，`train` 会自动完成最终预测，不需要运行中人工选择模型。
- Unseen Task 使用课程公布的原始发布时间计算 **24 小时**，不是从程序启动时重新计时；训练预算会缩短到剩余窗口，预测/评估输出前再次检查。

评分已改为：Released Tasks 性能 20%、Unseen Task 性能 10%、自主系统设计 30%、执行与可复现报告 10%、展示 30%。小组为 2–4 人。

## 2. API 与模型计费配置

默认使用规则规划器，无需 API。启用 LLM 时使用 `configs/final_policy.llm.example.json`，在 `configs/model_registry.json` 填写计费和程序运行需要的信息：

- `model_id`、`family`：请求模型名及 Qwen / DeepSeek 家族。
- `release_date`：实际发布日期，不得晚于 2026-10-21。
- `standard_pricing: true`：采用官方标准价格。
- `input_rmb_per_million`、`output_rmb_per_million`：截至 2026-10-21 的标准人民币单价；分档价格使用 `pricing_tiers`。
- `context_window_tokens`：调用前预算预留使用的上下文上限，是程序参数，不是作业证明材料。

顶层保留 `schema_version: 1`、`currency: "CNY"`、`pricing_as_of: "2026-10-21"` 和 `models` 列表。当前目录为空，不会编造日期或将未知价格填成零。

**无需填写 `reviewed`、`pinned`，无需提交证明文件、链接、截图或 SHA-256。** `response_model_ids` 可省略，默认响应模型名与请求名相同；平台返回其他名称时再配置。旧配置中可选的 `evidence` 如已提供，仍会检查其完整性，但不用为运行专门准备。

服务端需返回真实输入、输出 token 用量，推理输出也计入费用。程序保留每次调用与费用记录，单题预算不超过人民币 20 元。远程使用 HTTPS，密钥从环境变量读取。

## 3. 共享账本和保守预算

正式运行必须设置一个持久、绝对路径的 `SUPERAGENT_USAGE_ROOT`，所有 agent、所有启动和所有冻结副本使用同一路径。同一个课程任务使用同一 `evaluation_id`；该字段只用于费用归集，不传给规划器。

```powershell
# 正式运行前设置为整个小组共用的持久目录。
$env:SUPERAGENT_USAGE_ROOT = (Join-Path (Get-Location) '.model_usage')
# 如启用远程 LLM，再以安全方式设置 HKUST_API_KEY，勿把密钥写进提交文件。
```

账本目录按任务 ID 的 SHA-256 分开，记录 agent ID、运行 ID、调用 ID、模型身份、实际 token、原始 usage、适用价格档位、费用和请求路径。每次调用前用文件锁检查额度并写入 pending 预留；当前实现串行发起 LLM 请求，不支持同一账本上的并行在途请求。

预留按完整上下文上限乘以最大输入/输出标准价格计算，覆盖推理输出；成功返回后按实际 usage 结算并释放差额。这比实际请求的费用保守，特别是超长上下文模型，可能导致原本便宜的请求被拒绝；拒绝时不发请求，自动继续规则搜索。`max_tokens` 控制请求长度，但不是预算账本唯一依赖的依据。

HTTP 错误、超时、缺失 usage、无法核对模型身份或异常 token 总量都不能被记作零。程序保留预留，标记 `usage_unknown` 或核账异常并停止该次运行，不自动重试；后续启动也会因未核清的共享账本而拒绝继续调用。此类结果不能被当作费用已合规的完整运行。进程崩溃可能留下锁或 pending，请保留原始账本和供应商记录供核对，不要删除账本重置预算；正式自主运行中不加入人工修复继续执行的步骤。

这些措施依赖核验过的模型目录和服务端准确的模型/用量报告；它们不是对供应商实际后台实现或跨机器独立调用的认证。所有组内调用必须接入同一共享账本，提交时连同共享账本原件与每次运行的导出快照一起提供。

## 4. 固定策略并创建提交版本

先完成开发和允许的复验，核对任务要求及全部配置。`configs/final_policy.json` 是不调用 LLM 的策略模板；要用 LLM 则先完成上一节再选择 LLM 模板。

```powershell
# 以下为正式提交路径示例；开发验证使用单独的 runs/ 目录。
python main.py freeze --policy configs/final_policy.json --output releases/final
python releases/final/main.py verify-freeze --manifest releases/final/frozen_manifest.json
```

冻结包包含程序、依赖说明、配置/文档/检查源码、完整算法政策，以及启用 LLM 时的模型计费配置。记录每个文件的 SHA-256 及运行代码指纹。此命令不训练模型、不调用 LLM；开发验证生成的冻结副本不等于课程最终提交版本。

正式 `released` / `unseen` 配置必须引用 `frozen_manifest.json`。程序校验包内文件、运行代码和策略；不能在任务配置中覆盖冻结的搜索次数、随机种子、训练参数、LLM 身份、价格目录等。任务可以填写数据路径、模态列、标签列、指标等课程规定的输入信息。不要通过把正式运行标成 development 绕开冻结流程。

任务配置应放在冻结包之外，按课程公布的数据 schema 填写；不要编辑冻结包里的示例文件，也不要借输入字段做人工特征筛选或调参。固定源码不保证远程 LLM 响应或不同硬件上位级一致，因此必须保存完整响应、环境与实验轨迹。

可继续在工作区开发其他版本，但旧冻结包不能修改。后续换策略或代码意味着新版本，报告中的 Released Tasks 需要按同一个最终提交版本重新自主生成结果。哈希清单不是外部签名；应将最终清单与课程提交版本一起保存，以供核对。

## 5. Released Tasks 与 Unseen Task

课程任务发布后，按官方格式和指标填写 `configs/released.example.json` 或 `configs/unseen.example.json`。示例 ID 和时间戳是占位符，不能直接作为正式任务信息；代码没有自动猜测任何截止日期。

```powershell
python releases/final/main.py train --config configs/released.example.json --output runs/released_task_01
# 两个 section 完成展示、正式发布 Unseen Task 后，先填官方发布时间及数据路径：
python releases/final/main.py train --config configs/unseen.example.json --output runs/unseen_task_01
```

正式运行若给出 `test_csv`，完成最终自动选型和重训后，会自动生成 `predictions.csv` 和类别顺序说明，不用人为指定最佳模型。测试集不进入训练或规划上下文，程序也不会在搜索期间读取测试分数。测试标签若已公开，最终评估仍是独立 `evaluate` 操作，结果不能再反馈给模型选择。

Unseen 的训练预算受官方发布时间起 24 小时的剩余时间约束；保存预测/评估前再次检查期限。父进程的读盘、网络读取与推理不是 OS 级硬实时隔离，不能保证每个计算操作在窗口边界瞬时终止；到期后会拒绝产生新的预测/评估输出。课程提交和上传仍需小组在官方截止时间内完成，程序不会替组员提交。

期间不得人工改代码、修改尝试顺序、挑模型或改变冻结策略。程序失败时保留日志和已完成候选；若已有模型但后续输出失败，则标为 partial；账目异常的模型标为 unverified，不能声称合规完成。

## 6. 日志、报告与提交时点

每项任务保留：`model.joblib` 或明确标识的部分结果、预测文件、源码快照、数据哈希与划分、实验日志、`model_usage.json`、`task_model_calls.json`。启用 LLM 时另有 `model_calls.jsonl`、请求/原始响应、模型计费配置副本，并保留 `SUPERAGENT_USAGE_ROOT` 的共享账本原件。

最终提交包含统一程序、依赖和 README、Released Tasks 的模型与结果、`report.pdf`、执行与资源记录、AI 使用声明。报告成绩必须来自同一最终版本的完整自主运行；失败任务说明阶段、原因和有效部分结果。10 月 18 日的历史开发结果不能移作新版本正式成绩。

Final Submission 截止为所在 section **最后一周第一次课前一晚 23:59**；具体日期需按 L01/L02 课表确认。展示在最后一周各自课时进行。两个 section 都完成展示后，Unseen Task 发布，小组在 **24 小时内**用冻结版本运行并提交预测和日志，**无需新增报告**。

首轮冻结流程检查发现预测完成后的日志字段冲突，已在 agent.py 中修复；失败现场保留在 `runs/frozen_workflow_smoke_20261023/`。修复后的冻结流程检查位于 `runs/frozen_workflow_final_20261023/verification.json`，Released 与 Unseen 均从冻结入口自主训练并为无标签输入自动生成 114 行预测，账本与版本核验通过。模拟发布时间仅用于检查，不是官方 Unseen 发布时间。

修复后同一源码的三模态开发验证位于 `runs/development_full_requirements_final/`；此前 smoke 目录属于修复前版本，保留作排错历史。所有课程任务仍需使用实际课程数据、公告与最终提交版本重新执行。

## 学校 API 连通性

本地密钥文件提供访问凭据，API 接入示例提供 URL、Bearer 认证方式和请求模型名。这两项足够完成接入配置；不需要把原始密钥文件或个人测试脚本复制到仓库。`configs/final_policy.llm.example.json` 已填入确认过的学校参数：

- `llm_base_url`：`https://aigc-api.hkust-gz.edu.cn/api/v2`。规划器会自动追加 `/chat/completions`，不要重复填写该后缀。
- `llm_model_id`：`DeepSeek-V4-Pro`，这是学校接口接受的请求名。
- `llm_api_key_env`：`HKUST_API_KEY`，HTTP 使用 `Authorization: Bearer ...`。

这两个文件足够完成 API 接入，不需要另填版本证明或价格证据附件。正式计费仍需填写真实发布日期、标准单价和预算预留所用的上下文上限；学校余额或赠送额度不能替代作业的标准价格核算。

学校 API 的聊天补全接口为 `https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions`，模型名为 `DeepSeek-V4-Pro`。2026-10-23 独立开发探测返回 HTTP 200 / `OK`，实际输入 88、输出 13、总计 101 tokens。模型目录中的日期和价格不能仅从响应推断。此次调用不进入任何课程题目的运行结果，独立记录中的人民币费用为未知，而非零。

如代理影响学校内网接口连接，可按所在网络要求为该域名配置 `NO_PROXY`。保持 TLS 证书验证启用；无需更改全局代理。

```powershell
# 在项目根目录、已激活的 Python 环境中执行；输入现有密钥文件的路径。
$credentialPath = Read-Host '本地 API 密钥文件路径'
$env:HKUST_API_KEY = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $credentialPath).Path).Trim()
# 仅在网络要求直连学校接口时设置；保留原有其他直连域名。
$env:NO_PROXY = (@($env:NO_PROXY, 'aigc-api.hkust-gz.edu.cn') | Where-Object { $_ }) -join ','
# 输出目录须为新目录；测试会真实消耗 token。
python scripts/probe_hkust_api.py --output runs/hkust_probe_new
# 完成本次调用后可清除当前会话中的凭据。
Remove-Item Env:HKUST_API_KEY
```

也可以直接向探测脚本传入 `--key-file $credentialPath`，无需先设置环境变量；完整规划器只读取配置指定的环境变量。不要输出密钥值、把密钥写入 JSON，或将密钥文件提交到 Git。正式运行前还需设置持久的共享 `SUPERAGENT_USAGE_ROOT` 并完成计费配置和冻结步骤。

探测脚本只从环境变量或指定的现有文件读取凭据，不复制密钥；请求和响应摘要分别保存，不记录 Authorization。它仅检查开发连通性，不绕过正式运行的模型目录、预算账本与证据要求。开发探测记录保存在本地 runs/ 目录，不纳入 Git 仓库；未知用量和费用不能假定为零。
