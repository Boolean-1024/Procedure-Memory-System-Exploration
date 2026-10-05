# 记忆系统评测 —— 实现说明与运行指南

代码基于 [EvoMemBench](https://github.com/DSAIL-Memory/EvoMemBench)（clone 到 `EvoMemBench/`），按照"训练集建库 → 测试集只读"的协议改造。
已完成：**ALFWorld（主实验）**、**BFCL 多轮工具调用**。AppWorld（可选）已下载数据，还没接入记忆系统。
MemP 按指示用 ACE 代替，没有实现。

## 0. 环境（已装好）

| 组件 | 位置 | 说明 |
|---|---|---|
| 评测客户端 venv（Windows, Python 3.11） | `D:\Project\Thesis_Claude\.venv` | 安装脚本 `setup/win_client_setup.sh`；包含 agentenv、BFCL、mem0、A-mem、MemoryOS |
| ALFWorld 环境服务器（WSL Ubuntu, Python 3.9） | `~/alfworld-server/.venv`，数据在 `~/.cache/alfworld` | 安装脚本 `setup/wsl_alfworld_setup.sh`。TextWorld 不支持原生 Windows，所以放在 WSL |

### 大模型接口

火山引擎 Ark SDK 已全部替换成 OpenAI 兼容接口。密钥和模型从环境变量读取，**配置文件里不写密钥**：

| 变量 | 默认值 / 回退 |
|---|---|
| `LLM_API_KEY` | → `OPENAI_API_KEY`（Windows 系统级变量也能读到） |
| `LLM_BASE_URL` | → `OPENAI_BASE_URL` → `https://api.openai.com/v1` |
| `LLM_MODEL` | `gpt-4.1-mini`（agent 和记忆侧 LLM 用同一个模型） |
| `EMBED_API_KEY` / `EMBED_BASE_URL` | → 同上 |
| `EMBED_MODEL` | `text-embedding-3-small` |

为什么默认用 `gpt-4.1-mini` 而不是 gpt-5 系列：A-mem、MemoryOS 的代码里写死了 `temperature` 和 `max_tokens`，gpt-5 系列会拒绝这两个参数。我加了兼容补丁（`_llm.py` / `llm_env.py` 里的 `install_openai_hooks`），会自动删掉 `temperature`，并把 `max_tokens` 改成 `max_completion_tokens`，所以换成 gpt-5 也能跑。但推理模型在 2,420 局 × 20 轮的规模下又慢又贵。

换成其他 OpenAI 兼容服务（比如 DeepSeek）的例子：

```powershell
$env:LLM_BASE_URL="https://api.deepseek.com/v1"; $env:LLM_API_KEY="..."; $env:LLM_MODEL="deepseek-chat"
```

---

## 1. ALFWorld（主实验）

**协议**（`scripts/run_alfworld_main.py`）：
1. `rollout`：不带记忆的 agent 跑一遍 **200 局训练任务**，保存轨迹和 reward。这 200 局从 AgentGym 训练映射（编号 0–2419，`json_2.1.1/train`）中按任务类型分层抽取（seed 0），各类型局数与测试集相同（46/45/37/28/25/19），规模与 EvoMemBench 的 200 局一致。加 `--train_subset 0` 可用全部 2,420 局，`--train_dist train` 则按训练集原比例抽样。
2. `build`：**每个记忆系统都读入同一批轨迹**，通过它自己的 `update(conversation, idx, reward)` 写入记忆（`scripts/build_memory_from_trajectories.py`）。这样不同系统之间没有 rollout 方差。每个系统建一个共享库（不按任务类别分库）。
3. `test`：在测试集（编号 2420–2619，`json_2.1.1/valid_train`）上以 `--readonly_memory` 运行；无记忆基线跑同样的 200 局。
4. `report`：输出总成功率、6 个任务类型各自的成功率，以及 token 成本，写到 `report.md` / `report.csv`。

系统列表：`awm, reasoning_bank, ace, mem0, amem, memoryos, bm25, vector`（`vector` 就是原样保存轨迹的向量检索，对应原来的 `qwen3_embedding` 适配器），外加 `no_memory`。
超参数沿用 EvoMemBench 原脚本：AWM top3、RB top1、mem0/amem/memoryos top3、BM25/向量 top10 加 1024-token 分块。

### 运行

```powershell
# 终端 1：启动环境服务器（一直开着）
wsl -d Ubuntu -- bash /mnt/d/Project/Thesis_Claude/setup/wsl_alfworld_server.sh 36005
```

```powershell
# 终端 2
cd D:\Project\Thesis_Claude\EvoMemBench\Cross-Episode-Execution\Embodied-AI\CROSSEP-EMB
$env:PYTHONUTF8=1
..\..\..\..\.venv\Scripts\python.exe scripts\run_alfworld_main.py all
```

也可以分步运行（每一步都能断点续跑，已完成的局或轨迹会自动跳过）：

```powershell
python scripts\run_alfworld_main.py rollout --parallel 8
python scripts\run_alfworld_main.py build   --systems ace awm reasoning_bank
python scripts\run_alfworld_main.py test    --systems no_memory ace awm reasoning_bank
python scripts\run_alfworld_main.py report
```

输出目录默认是 `CROSSEP-EMB/output/main_<model>/`：

```text
train_rollouts/alfworld_<idx>.json      # 共享训练轨迹（conversations + reward）
configs/<system>.json                   # 每个系统的配置（不含密钥）
memory/<system>/...                     # 记忆库
memory/<system>/build_log.jsonl         # 写入日志（每条轨迹一行）
memory/<system>/build_summary.json      # 建库的 token 和耗时
test/<system>/alfworld_<idx>.json       # 测试轨迹
test/<system>/retrieval_log.jsonl       # 检索日志（每局一行）
report.md / report.csv
```

### 日志格式

- **写入日志** `build_log.jsonl`：`{"event":"update","data_idx","reward","stats":{tokens},"llm_calls":[{prompt_tail, output}],"store_files":{path:{files,bytes}}}`
  `llm_calls[].output` 是记忆侧 LLM 实际产出的内容：AWM 归纳出的 workflow、ACE 的 reflector/curator 输出、RB 抽取的条目、mem0 的事实等。
- **检索日志** `retrieval_log.jsonl`：`{"event":"inject","data_idx","query","injected","injected_chars","stats"}`
  `injected` 就是这一局实际拼进 system prompt 的记忆原文。

---

## 2. BFCL 多轮工具调用

**数据划分**：BFCL 没有官方训练集。这里把 4 个环境（gorilla_fs / vehicle_control / trading_bot / travel_api，每个 50 条）各自用固定种子按 1:1 划分，得到每个环境 25 条训练、25 条测试，总共 100/100。划分写在 `split.json` 里，之后的运行都复用这个文件。
**论文中可以这样描述**："Since BFCL has no official training split, we randomly split each of the four environments 1:1 (seed 0), yielding 25 train / 25 test samples per environment; memory is built per environment from the train half and evaluated read-only on the test half."

**协议**（`bfcl_eval/scripts/cross_episode/run_split_experiment.py`）和 ALFWorld 一样：
1. 无记忆 agent 跑一遍训练集（FC 模式）。
2. 每个系统、每个环境各建一个新库，读入该环境的训练轨迹（`full_message_history`）。
3. 只读测试。
4. 出报告（成功率、progress，按环境分别统计）。

注意：BFCL 的 `update()` 接口本来就不传 reward。AWM 和 RB 这类需要成功信号的系统，用的是原实现里的 LLM judge。

```powershell
cd D:\Project\Thesis_Claude\EvoMemBench\Cross-Episode-Execution\Tool-Using\CROSSEP-TOOL
$env:PYTHONUTF8=1
..\..\..\..\.venv\Scripts\python.exe -m bfcl_eval.scripts.cross_episode.run_split_experiment all
```

输出目录默认是 `CROSSEP-TOOL/cross_episode_results/split_main_<model>/`：

```text
split.json
train_rollouts/result.jsonl
memory/<system>/<env>/build_log.jsonl
memory/<system>/<env>/build_summary.json
test/<system>/<env>/result.jsonl
test/<system>/<env>/per_sample.csv
test/<system>/<env>/memory_log.jsonl      # 检索日志（utilize 返回的 snippet）
report.md / report.csv
```

---

## 2.5 AppWorld（已下载，还没接入记忆系统）

- 代码：`ace-appworld/`（ACE 官方的 AppWorld 分支，必须从源码安装），环境装在 WSL 的 `~/appworld-venv`（Python 3.11）。安装脚本是 `setup/wsl_appworld_setup.sh`，里面把 click 锁在 8.2 以下，否则 `appworld` 命令行会报错。
- 数据：`ace-appworld/data/`。实际划分是 **train 90 / dev 57 / test_normal 168 / test_challenge 417**（数的是 `data/datasets/*.txt` 的行数）。论文里常见的 105/60 是早期版本的数字。
- 冒烟测试：`setup/appworld_smoke.sh`，能加载 train 第一个任务并执行 API。

## 3. 对原代码的改动

| 改动 | 文件 |
|---|---|
| 新增统一的 LLM/Embedding 客户端：环境变量解析、gpt-5 兼容、线程内 LLM 输出捕获、关闭 mem0 遥测 | `CROSSEP-EMB/scripts/memory/_llm.py`，`CROSSEP-TOOL/bfcl_eval/memory/llm_env.py` |
| `Ark(...).batch.chat.completions` 改成 `OpenAI(...).chat.completions` | ALFWorld：`ace_adapter.py`、`reasoning_bank_adapter.py`、`mem0_adapter.py`、`_ark_utils.py`；BFCL：`ark_client.py`、`amem_backend.py` |
| DashScope 写死的配置改成从环境变量读取 | `amem_adapter.py`、`memoryos_adapter.py`、`reasoning_bank_adapter.py`、`qwen3_embedding_adapter.py`、AWM provider、BFCL `memoryos_backend.py`、`EvolveLab/providers/_dashscope_embedder.py` |
| 写入和检索日志的包装器 | `scripts/memory/logging_wrapper.py`，`bfcl_eval/memory/logging_wrapper.py` |
| ALFWorld 评测脚本：新增 `--api_mode openai`（默认）、`--temperature`、`--memory_log`；`.env` 改为可选，且不覆盖已有的环境变量 | `scripts/eval_alfworld_with_memory.py` |
| BFCL 生成脚本：OpenAI 接口、`--model`、`--memory-log`；把每个后端的 kwargs 抽成 `memory_kwargs()`；`--ids` 为空时直接报错（原来会悄悄跑全部 200 条） | `run_batch_generate.py` |
| 新增的入口脚本 | `run_alfworld_main.py`、`build_memory_from_trajectories.py`、`run_split_experiment.py` |
| 全局限流重试：429 和临时性错误按指数退避重试，最多 30 次（`LLM_RETRY_ATTEMPTS`）。原因是 mem0 和 A-mem 会吞掉异常、静默跳过写入 | `_llm.py` / `llm_env.py` 中的 `_with_retry` |
| A-MEM（ALFWorld）笔记修复：Task 字段改用观察中的 "Your task is to …" 那句，原来取的是 system prompt 前 300 字符 | `amem_adapter.py` |
| AWM（BFCL）只学成功轨迹：建库时把训练集的评测结果传给 `metadata.is_correct`，原来默认全部当作成功 | `run_split_experiment.py` |
| mem0（ALFWorld）embedding 输入截断到 8000 token（原来 6/200 条轨迹报 400 错误），与 BFCL 后端的做法一致 | `mem0_adapter.py` |
| A-mem 的 `max_tokens` 改为 16384（BFCL 后端原值 4096，24/96 次 JSON 被截断） | A-mem `llm_controller.py`、BFCL `amem_backend.py` |
| 新增参数：ALFWorld `--train_subset`（默认 200）/ `--train_dist`；BFCL `--test_per_env` | 两个入口脚本 |
| 建库审计脚本，检查静默失败 | `setup/audit_builds.py` |
| 第三方库的小修 | MemoryOS：所有 `text-embedding-*` 都走 API，并且**不再把 API key 打印到 stdout**；A-mem：`max_tokens` 从 1000 改成 4096（原值会截断 memory-evolution 的 JSON，导致 evolution 静默失效） |

可以用 `cd EvoMemBench && git diff` 查看全部改动（新增文件用 `git status` 查看）。

## 4. 已知问题和成本

**Pilot 结果**（2026-10-03，gpt-4.1-mini，6 条训练轨迹，6 局测试，目录 `D:\Project\Thesis_Claude\scratch_alf_pilot`）：9 个配置全部跑通，0 个错误，写入日志和检索日志都正常。样本量太小，成功率没有统计意义。

**ALFWorld 成本估算（默认 200 局训练 + 200 局测试，gpt-4.1-mini）**：

| 阶段 | 估算 |
|---|---|
| rollout：200 局 × 约 5 万 token | 约 1,000 万 token，约 $4 |
| build（不含 MemoryOS）：每条轨迹约 1 万 token | 每个系统约 200 万 token，不到 $1 |
| build MemoryOS：每条轨迹约 6.9 万 token、约 108 秒 | 约 1,400 万 token，约 $7，串行约 6 小时 |
| test：200 局 × 9 个配置；BM25 和向量检索每局约 20 万 token | 约 1.5 亿 token，约 $60 |

- **MemoryOS 建库慢**：它会把每条轨迹拆成约 20 个 (obs, action) 对，逐个调用 `add_memory`，短期记忆满了就调 LLM 做摘要。每条轨迹约 50 次 LLM 调用、约 108 秒；200 条约 6 小时，全量 2,420 条约 73 小时。这是算法本身的特性，我没有改。
- mem0、A-mem、MemoryOS 在测试时默认串行运行（`SERIAL_TEST`），因为它们读操作的线程安全性没有验证过。加上 `--force_parallel` 可以并行。
- **ReasoningBank 不看环境 reward**：原实现总是用 LLM-as-judge 判断一条轨迹是成功还是失败（这与 ReasoningBank 论文的设定一致），所以它的 success/fail 标签可能和环境的 reward 不一样（pilot 中 6 条里有 1 条不一致）。AWM 则只用 reward≥1 的轨迹。论文里如果要对比，最好说明这一点。
- 建库默认 `--build_parallel 1`。ACE 的 playbook、AWM 这类有状态的系统需要按顺序写入，结果才可复现。
