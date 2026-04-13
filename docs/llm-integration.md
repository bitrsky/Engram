# Engram LLM 集成指南

> **代码示例驱动** — 本文档以完整代码示例为主，展示 Engram 的 LLM 集成点。

## 目录

1. [架构概览](#1-架构概览)
2. [LLM 回调协议](#2-llm-回调协议)
3. [集成点一：事实提取](#3-集成点一事实提取)
4. [集成点二：搜索重排](#4-集成点二搜索重排)
5. [集成点三：查询重写](#5-集成点三查询重写)
6. [集成点四：时间推理](#6-集成点四时间推理)
7. [与 Pattern Learning 的关系](#7-与-pattern-learning-的关系)
8. [配置说明](#8-配置说明)
9. [错误处理与降级策略](#9-错误处理与降级策略)
10. [成本估算](#10-成本估算)
11. [端到端示例](#11-端到端示例)

---

## 1. 架构概览

Engram 有 **两种** LLM 集成模式：

```
模式 A：回调协议（推荐）
┌─────────────────────┐          ┌──────────────────────┐
│  EchoBot/echo-code   │ ◄──────  │       Engram          │
│  (本身就是 LLM)      │  llm_fn  │  (记忆库，不调 API)    │
│                      │  回调    │                        │
│  执行 LLM 推理       │ ◄──────  │  构建 prompt + 解析    │
└─────────────────────┘          └──────────────────────┘

模式 B：内置 HTTP（向后兼容）
┌──────────────────────┐         ┌─────────────────────┐
│       Engram          │ ──────► │  LLM API            │
│  (通过 urllib 调用)   │  HTTP   │  ollama / openai /   │
│                      │         │  anthropic            │
└──────────────────────┘         └─────────────────────┘
```

**优先级**：`llm_fn 回调` > `内置 HTTP` > `启发式规则`

四个 LLM 集成点的数据流：

```
用户查询
  │
  ▼
┌────────────────────┐
│ ① 查询重写         │  "what auth?" → "authentication provider Clerk Auth0"
│   (llm.rewrite)    │  默认关闭，需 query_rewrite: true
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  向量搜索           │  ChromaDB → top-20 候选
│  (index.vector)    │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ ② 搜索重排          │  top-20 → LLM 打分 → top-5
│   (llm.rerank)     │  默认开启（当 LLM 可用时）
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ ③ 时间推理          │  "when did we switch?" + 记忆时间戳 → 日期推理
│   (llm.temporal)   │  仅在检测到时间标记词时触发
└────────┬───────────┘
         │
         ▼
    搜索结果返回

记忆写入
  │
  ▼
┌────────────────────┐
│ ④ 事实提取          │  "We switched to Clerk" → (saas-app, uses, Clerk)
│   (llm.extract)    │  自动提取三元组事实
└────────────────────┘
```

---

## 2. LLM 回调协议

### 协议定义

```python
# engram/llm.py

@runtime_checkable
class LLMCallback(Protocol):
    """LLM 调用协议。

    宿主 agent 提供实现，将 prompt 发送给自己使用的模型。
    Engram 只关心接口签名。

    Keyword arguments 可包含:
        temperature (float): 采样温度
        max_tokens (int):    最大响应 token 数
        timeout (int):       超时秒数
    """
    def __call__(
        self,
        prompt: str,
        system: str = "",
        **kwargs,
    ) -> Optional[str]: ...
```

### 注入方式

**方式一：通过 MemoryStack 构造函数**

```python
from engram.layers import MemoryStack
from engram.config import EngramConfig

def my_llm(prompt: str, system: str = "", **kwargs) -> str:
    """宿主 agent 的 LLM 实现。"""
    return call_my_model(prompt, system_message=system)

config = EngramConfig()
stack = MemoryStack(config=config, llm_fn=my_llm)

# 现在 stack 的所有操作会使用 my_llm 进行 LLM 推理
context = stack.recall(message="auth provider decision")
```

**方式二：通过 MCP Server 全局注入**

```python
from engram.mcp_server import set_llm_callback

def my_llm(prompt: str, system: str = "", **kwargs) -> str:
    return call_my_model(prompt, system_message=system)

# 注入后，所有 MCP tool 调用自动使用此回调
set_llm_callback(my_llm)
```

**方式三：函数级传递**

```python
from engram.search import search
from engram.remember import remember
from engram.extract import extract_facts

# 搜索时传入
results = search(query="auth", index_manager=idx, config=cfg, llm_fn=my_llm)

# 记忆写入时传入
result = remember(content="We use Clerk", project="saas-app", config=cfg, llm_fn=my_llm)

# 事实提取时传入
facts = extract_facts(content="We use Clerk", project="saas-app", config=cfg, llm_fn=my_llm)
```

### 优先级规则

每个支持 LLM 的函数都遵循相同的三级降级逻辑：

```python
def extract_facts(content, project=None, existing_facts=None, config=None, llm_fn=None):
    config = config or EngramConfig()

    # 优先级 1：外部回调（推荐）
    if llm_fn is not None:
        try:
            return extract_facts_via_callback(content, llm_fn, project, existing_facts)
        except Exception:
            pass  # 降级到下一层

    # 优先级 2：内置 HTTP LLM（向后兼容）
    if config.llm_available:
        try:
            return extract_facts_llm(content, project, existing_facts, config)
        except Exception:
            return extract_facts_heuristic(content, project)

    # 优先级 3：启发式规则（始终可用）
    return extract_facts_heuristic(content, project)
```

---

## 3. 集成点一：事实提取

### 功能说明

从文本中提取结构化事实（三元组：subject → predicate → object）。

### Prompt 构造

```python
# engram/llm.py — _EXTRACT_SYSTEM + _EXTRACT_PROMPT

system = "You are a fact extraction engine. Output only valid JSON."

prompt = f"""Extract structured facts from the following text.

Project context: {project}
{known_facts_section}       # ← 已知事实（用于冲突检测）
Text to analyze:
---
{content}
---

Extract ALL factual claims as (subject, predicate, object) triples.
Include: decisions, assignments, technical choices, timelines, concerns, metrics, relationships.
Skip: opinions about code quality, generic statements, pleasantries, hypotheticals.

For each fact:
- subject: the entity (person, project, component)
- predicate: the relationship (uses, assigned_to, decided, status, etc.)
- object: the value
- confidence: 0.0-1.0 based on how certain this fact is from the text
- temporal: ISO date or relative time phrase if present, empty string otherwise
- conflicts_with: "subject → predicate → object" of a conflicting known fact, or empty string

Output ONLY a valid JSON array, no markdown formatting:
[
  {{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9, "temporal": "", "conflicts_with": ""}}
]"""
```

### 已知事实注入

当存在已知事实时，会在 prompt 中注入上下文（最多 20 条）：

```python
known_facts_section = ""
if existing_facts:
    facts_lines = []
    for f in existing_facts[:20]:
        facts_lines.append(f"  - {f.subject} → {f.predicate} → {f.object}")
    known_facts_section = (
        "\nKnown facts about this project:\n"
        + "\n".join(facts_lines)
        + '\n\nIf any extracted fact contradicts a known fact above, '
        'note it in "conflicts_with".\n'
    )
```

### 响应解析

`_parse_llm_response()` 处理多种 LLM 输出格式：

```python
def _parse_llm_response(response_text: str) -> List[FactCandidate]:
    text = response_text.strip()

    # 1. 去除 markdown 代码块包装: ```json ... ```
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    # 2. 定位 JSON 数组（跳过前后文本）
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start != -1 and bracket_end > bracket_start:
        text = text[bracket_start : bracket_end + 1]

    # 3. 修复常见 JSON 问题（尾逗号）
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # 4. 解析 → FactCandidate 列表
    parsed = json.loads(text)
    ...
```

### 回调 vs 内置 HTTP 对比

| 方面 | `llm_fn` 回调 | 内置 HTTP |
|------|---------------|-----------|
| 调用方 | 宿主 agent | engram 自己 |
| 模型选择 | 宿主 agent 决定 | `config.toml` 配置 |
| 认证 | 宿主 agent 处理 | 需配置 `api_key` |
| System 消息 | 通过 `system` 参数传递 | 硬编码在 HTTP 请求中 |
| 温度 | 通过 `**kwargs` 传递 | 硬编码 `0.1` |

---

## 4. 集成点二：搜索重排

### 功能说明

两阶段检索：先用向量搜索获取 top-N 候选，再用 LLM 按相关性重排。

### Prompt 构造

```python
# engram/llm.py — _RERANK_SYSTEM + rerank_with_llm()

system = "You are a search relevance judge. Output ONLY a JSON array of document numbers (1-based), most relevant first."

prompt = f"""Given a question and a list of documents, rank them by relevance.

Return ONLY a JSON array of document numbers (1-based), most relevant first.
Return the top {top_k} most relevant documents.

Example output: [3, 1, 7, 5, 2]

Question: {query}

Documents:
[1] We decided to use Clerk for authentication after comparing Auth0...
[2] Database migration from MongoDB to Postgres completed last week...
[3] Sprint planning notes for Q1 2026...
...

Output (top {top_k} document numbers, JSON array):"""
```

### 候选文档截断

每个候选限制 300 字符，保持总 prompt 紧凑：

```python
_MAX_DOC_CHARS = 300

for i, hit in enumerate(candidates):
    content = hit.content.replace("\n", " ").strip()
    if len(content) > _MAX_DOC_CHARS:
        content = content[:_MAX_DOC_CHARS] + "..."
    doc_lines.append(f"[{i+1}] {content}")
```

### 响应解析与索引映射

```python
def _parse_rerank_indices(response, n_candidates, top_k):
    text = response.strip()

    # 策略 1：JSON 数组 — [3, 1, 5]
    json_match = re.search(r'\[[\d\s,]+\]', text)
    if json_match:
        arr = json.loads(json_match.group())
        indices = [int(x) - 1 for x in arr if 0 <= int(x) - 1 < n_candidates]
        return deduplicate(indices)[:top_k]

    # 策略 2：提取所有数字 — "Most relevant: 3, 1, 5"
    numbers = re.findall(r'\b(\d+)\b', text)
    indices = [int(n) - 1 for n in numbers if 0 <= int(n) - 1 < n_candidates]
    return deduplicate(indices)[:top_k]
```

**不足量回填**：如果 LLM 返回的排名数少于 `top_k`，用原始顺序中未出现的候选回填：

```python
# LLM 返回 [3, 1]，但 top_k=5
reranked = [candidates[2], candidates[0]]  # 来自 LLM
# 回填：candidates[1], candidates[3], candidates[4]
for i, c in enumerate(candidates):
    if i not in seen:
        reranked.append(c)
```

### 完整调用链

```
search() / layers.recall()
  │
  ├── config.rerank_enabled? ──No──► vector_search(n=5)
  │
  └── Yes
       │
       ▼
  index.vector_search_reranked()
       │
       ├── vector_search(n=20)           # Stage 1: 取 20 个候选
       │
       └── rerank(candidates, llm_fn)    # Stage 2: LLM 重排
            │
            ├── llm_fn provided? ──Yes──► llm.rerank_with_llm()
            │
            └── No ──► _call_llm_for_rerank() (内置 HTTP)
```

---

## 5. 集成点三：查询重写

### 功能说明

将模糊查询扩展为更具体的搜索关键词，提升向量搜索的召回率。

### Prompt 构造

```python
# engram/llm.py — _REWRITE_SYSTEM + _REWRITE_PROMPT

system = "You are a search query expander. Given a vague or short question, "
         "rewrite it as a more specific query with related keywords to improve "
         "semantic search recall. Output ONLY the expanded query — no explanation."

prompt = f"""Expand this question into a more specific search query with related terms.

Original question: {query}

Rules:
- Keep the original intent
- Add synonyms and related terms
- Keep it under 50 words
- Output ONLY the expanded query, nothing else"""
```

### 安全检查

```python
def rewrite_query(query: str, llm_fn: LLMCallback) -> str:
    result = llm_fn(prompt, system=system, temperature=0.0, max_tokens=128)
    if result and result.strip():
        expanded = result.strip().strip('"').strip("'")
        # 安全检查：太长或太短则放弃
        if len(expanded) > 500 or len(expanded) < 3:
            return query  # 回退到原始查询
        return expanded
    return query  # 失败回退
```

### 配置

```toml
# config.toml
[llm]
query_rewrite = true   # 默认 false — 增加约 200ms 延迟
```

```bash
# 环境变量
export ENGRAM_QUERY_REWRITE=1
```

---

## 6. 集成点四：时间推理

### 功能说明

检测时间类问题（"when did..."、"how long ago..."），结合记忆时间戳用 LLM 推理答案。

### 时间标记检测（零成本门控）

```python
_TEMPORAL_MARKERS = re.compile(
    r"\b("
    r"when did|when was|when were|when is|when are|"
    r"how long ago|how many days|how many weeks|how many months|how many years|"
    r"days ago|weeks ago|months ago|years ago|"
    r"since when|last time|first time|"
    r"before|after|during|between .+ and|"
    r"timeline|chronolog|sequence of events"
    r")\b",
    re.IGNORECASE,
)

def is_temporal_query(query: str) -> bool:
    return bool(_TEMPORAL_MARKERS.search(query))
```

仅在检测到标记词时才调用 LLM — 大部分查询不会触发。

### Prompt 构造

```python
system = "You are a temporal reasoning assistant. Given a time-related question "
         "and memory excerpts with timestamps, reason about the answer. "
         "Be precise about dates and durations. "
         "Output only the answer — no preamble."

prompt = f"""Question: {query}

Memory excerpts (with dates):
[2026-01-15] We decided to use Clerk for authentication after comparing Auth0...
[2026-02-20] Maya started the Clerk migration, estimated 2 weeks...
[2026-03-01] Auth migration completed, all users migrated to Clerk...

Based on these memory excerpts, answer the question about timing/dates.
If you cannot determine the answer, say "Unable to determine from available memories."
Answer:"""
```

### 结果附加

时间推理答案附加到第一个搜索结果的内容中：

```python
# search.py
if temporal_answer and enriched:
    enriched[0].content = (
        enriched[0].content.rstrip()
        + f"\n\n[Temporal Reasoning]\n{temporal_answer}"
    )
```

### 配置

```toml
# config.toml
[llm]
temporal_reasoning = true   # 默认 true（仅标记检测门控，成本极低）
```

---

## 7. 与 Pattern Learning 的关系

Pattern Learning（`learn.py`）不是 LLM 集成点，但它**观察**LLM 集成点（事实提取）的输出。

```
                                零额外 LLM 调用
                                ┌─────────────┐
   ④ 事实提取 (LLM)             │  learn.py    │
   输出: [FactCandidate]  ─────►│  比较:       │
                                │  LLM 提取了   │──► learned_patterns.toml
   ① 质量门控 (regex)           │  但 regex 没  │
   输出: matched_categories ───►│  匹配到 → 学! │
                                └─────────────┘
```

**关键设计**：
- Pattern Learning **复用**事实提取的 LLM 调用输出，不发起任何新的 LLM 调用
- 当 LLM 不可用时（纯启发式模式），Pattern Learning 仍然工作——只是学习机会更少（启发式提取的事实比 LLM 少）
- 学会的 pattern 可以**替代** LLM 做部分判断（如中文决策关键词匹配），降低对 LLM 的依赖

详见 [data-flow.md — Step 8: Pattern Learning](./data-flow.md)。

---

## 8. 配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ENGRAM_LLM_PROVIDER` | LLM provider: `ollama` / `openai` / `anthropic` / `none` | `none` |
| `ENGRAM_LLM_MODEL` | 模型名称 | 按 provider 默认 |
| `ENGRAM_LLM_API_KEY` | API 密钥 | `""` |
| `ENGRAM_LLM_BASE_URL` | API 基地址 | 按 provider 默认 |
| `ENGRAM_RERANK` | 启用重排 (`1`/`0`) | `1` (当 LLM 可用) |
| `ENGRAM_RERANK_CANDIDATES` | 重排候选数 | `20` |
| `ENGRAM_QUERY_REWRITE` | 启用查询重写 (`1`/`0`) | `0` |
| `ENGRAM_TEMPORAL_REASONING` | 启用时间推理 (`1`/`0`) | `1` |
| `ENGRAM_PROMOTION_THRESHOLD` | Pattern 学习晋升阈值 | `2` |

### config.toml

```toml
# ~/.engram/config.toml
[llm]
# rerank = true                  # LLM 重排（默认 true 当 think_fn 可用时）
# rerank_candidates = 20         # 重排候选数
# query_rewrite = false          # 查询重写（默认 false，增加延迟）
# temporal_reasoning = true      # 时间推理（默认 true）

[learning]
# promotion_threshold = 2        # 候选 pattern 晋升阈值（默认: 2）
```

### 功能矩阵

| 功能 | 需要 LLM | 默认状态 | 延迟影响 |
|------|----------|----------|----------|
| 事实提取 | 可选（有启发式） | 自动（有 LLM 用 LLM） | +200-500ms 写入时 |
| 搜索重排 | 是 | 开启 | +200-300ms 搜索时 |
| 查询重写 | 是 | **关闭** | +200ms 搜索时 |
| 时间推理 | 是 | 开启 | +200ms（仅时间查询） |
| Pattern 学习 | **否**（观察已有 LLM 输出） | 开启 | < 1ms（纯 Python） |

---

## 9. 错误处理与降级策略

### 降级总原则

**任何 LLM 失败都不应阻塞核心功能。** 每个集成点都有完整的降级路径：

```
┌──────────────────┐
│  llm_fn 回调      │──失败──┐
└────────┬─────────┘        │
         │ 成功              ▼
         │            ┌──────────────────┐
         │            │  内置 HTTP LLM   │──失败──┐
         │            └────────┬─────────┘        │
         │                     │ 成功              ▼
         │                     │            ┌──────────────────┐
         │                     │            │  启发式 / 跳过    │
         │                     │            └────────┬─────────┘
         ▼                     ▼                     ▼
     返回增强结果          返回增强结果          返回基础结果
```

### 各集成点降级策略

| 集成点 | LLM 失败时的行为 |
|--------|------------------|
| 事实提取 | 降级到正则启发式（~1-2 facts vs LLM 的 ~5-7） |
| 搜索重排 | 返回原始向量搜索排序（candidates[:top_k]） |
| 查询重写 | 使用原始查询（不扩展） |
| 时间推理 | 跳过，不附加时间答案 |

### 代码中的错误处理模式

```python
# 所有 LLM 集成点使用相同模式：
try:
    result = llm_fn(prompt, system=system, **kwargs)
    if result and result.strip():
        return parse_result(result)
except Exception as exc:
    logger.debug("LLM call failed: %s", exc)

return fallback_result  # 始终有合理的降级值
```

---

## 10. 成本估算

### Token 消耗预估

| 操作 | 输入 tokens (约) | 输出 tokens (约) | 频率 |
|------|-----------------|-----------------|------|
| 事实提取 | 200-400 | 100-200 | 每次 remember() |
| 搜索重排 (20 候选) | 800-1200 | 20-50 | 每次 search() |
| 查询重写 | 50-100 | 20-50 | 每次 search()（开启时） |
| 时间推理 | 300-500 | 50-100 | 仅时间查询 |

### 按模型费率估算

| 模型 | 每次 search() | 每次 remember() | 每月 (100次/天) |
|------|--------------|----------------|----------------|
| GPT-4o-mini | ~$0.0005 | ~$0.0002 | ~$0.60 |
| Claude Haiku | ~$0.0003 | ~$0.0001 | ~$0.36 |
| Ollama (本地) | $0 | $0 | $0 |
| **回调模式** | **宿主承担** | **宿主承担** | **—** |

> 使用回调模式时，LLM 成本由宿主 agent 统一承担，engram 不产生额外费用。

---

## 11. 端到端示例

### 示例：完整的记忆写入 + 搜索流程

```python
from engram.config import EngramConfig
from engram.layers import MemoryStack
from engram.remember import remember

# ── 定义 LLM 回调 ──────────────────────────────────────────
def my_llm(prompt: str, system: str = "", **kwargs) -> str:
    """使用 litellm 调用模型（示例）。"""
    import litellm
    response = litellm.completion(
        model="anthropic/claude-haiku",
        messages=[
            {"role": "system", "content": system} if system else None,
            {"role": "user", "content": prompt},
        ],
        temperature=kwargs.get("temperature", 0.1),
        max_tokens=kwargs.get("max_tokens", 1024),
    )
    return response.choices[0].message.content

# ── 初始化 ──────────────────────────────────────────────────
config = EngramConfig()
stack = MemoryStack(config=config, llm_fn=my_llm)

# ── 写入记忆 ────────────────────────────────────────────────
result = remember(
    content="We decided to switch from Auth0 to Clerk for authentication. "
            "Maya will lead the migration, estimated 2 weeks.",
    project="saas-app",
    topics=["auth", "migration"],
    memory_type="decision",
    config=config,
    llm_fn=my_llm,  # LLM 用于事实提取
)
# result.facts_extracted = 3
# 提取的事实:
#   saas-app → uses_auth → Clerk (confidence: 0.95)
#   Maya → assigned_to → auth migration (confidence: 0.9)
#   auth migration → estimated_duration → 2 weeks (confidence: 0.85)

# ── 搜索记忆 ────────────────────────────────────────────────
# L2: 上下文召回
context = stack.recall(message="how is the auth migration going?")
# 返回相关记忆（经 LLM 重排）

# L3: 深度搜索
results = stack.search("authentication provider decision")
# 返回格式化的搜索结果

# ── 时间查询 ────────────────────────────────────────────────
from engram.search import search as do_search

results = do_search(
    query="when did we decide to switch to Clerk?",
    index_manager=stack.index,
    config=config,
    llm_fn=my_llm,  # LLM 用于时间推理
)
# results.hits[0].content 末尾附加:
# [Temporal Reasoning]
# Based on the memory dated 2026-01-15, the decision to switch
# from Auth0 to Clerk was made on January 15, 2026.
```

### 示例：MCP Server 集成

```python
from engram.mcp_server import set_llm_callback, mcp

# 在 MCP server 启动时注入 LLM 回调
def agent_llm(prompt: str, system: str = "", **kwargs) -> str:
    # 宿主 agent 的 LLM 调用实现
    return call_agent_model(prompt, system)

set_llm_callback(agent_llm)

# 之后所有 MCP tool 调用自动使用此回调:
# - engram_search → 自动重排 + 可选查询重写 + 可选时间推理
# - engram_remember → 自动 LLM 事实提取
# - engram_recall → 自动重排
```

### 示例：不使用 LLM（纯启发式模式）

```python
from engram.config import EngramConfig
from engram.layers import MemoryStack

# 不注入 llm_fn，不配置 LLM provider
config = EngramConfig()  # llm.provider: "none"
stack = MemoryStack(config=config)

# 一切正常工作，只是：
# - 事实提取使用正则启发式（较少的提取结果）
# - 搜索无重排（纯向量相似度��序）
# - 无查询重写
# - 无时间推理
```

---

## 附录：文件变更总结

本次 LLM 集成涉及的代码变更：

| 文件 | 操作 | 变更说明 |
|------|------|----------|
| `src/engram/llm.py` | **新建** | LLM 回调协议 + 4 个 prompt 构建器 |
| `src/engram/learn.py` | **新建** | 自适应 pattern 学习引擎（观察 LLM 输出，零额外调用） |
| `src/engram/config.py` | 修改 | TOML 配置、三层 pattern 合并、`promotion_threshold`、`reload_learned_patterns()` |
| `src/engram/quality.py` | 修改 | 新增 `quality_gate_detailed()` 返回 matched_categories |
| `src/engram/extract.py` | 修改 | `extract_facts()` 接受 `llm_fn` 参数 |
| `src/engram/rerank.py` | 修改 | `rerank()` 接受 `llm_fn` 参数 |
| `src/engram/search.py` | 修改 | 集成查询重写 + 时间推理 + `llm_fn` 传递 |
| `src/engram/layers.py` | 修改 | `MemoryStack.__init__()` 接受 `llm_fn` |
| `src/engram/remember.py` | 修改 | `remember()` 接受 `llm_fn`，hook `learn_from_extraction()` |
| `src/engram/index.py` | 修改 | `vector_search_reranked()` 接受 `llm_fn` |
| `src/engram/mcp_server.py` | 修改 | 新增 `set_llm_callback()` 全局注入 |
| `src/engram/__init__.py` | 修改 | 导出 `LLMCallback` |
| `tests/test_llm.py` | **新建** | 36 个测试覆盖所有 LLM 功能 |
| `tests/test_learn.py` | **新建** | 41 个测试覆盖 pattern learning 全流程 |
| `tests/test_rerank.py` | 修改 | 新增 2 个 `llm_fn` 回调路径测试 |
