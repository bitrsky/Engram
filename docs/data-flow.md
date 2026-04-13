# Engram 完整数据流分析

> **Engram** — Markdown-First AI Memory System
>
> 核心理念：**Markdown is the source of truth**。索引（ChromaDB + SQLite）是派生物，可以随时删除重建。

---

## 目录

1. [系统架构总览](#1-系统架构总览)
2. [文件系统结构](#2-文件系统结构)
3. [写入路径（Write Path）](#3-写入路径write-path)
4. [读取路径（Read Path）](#4-读取路径read-path)
5. [索引层详解](#5-索引层详解)
6. [后台维护（Decay Engine）](#6-后台维护decay-engine)
7. [LLM 集成点](#7-llm-集成点)
8. [CLI 入口](#8-cli-入口)
9. [端到端数据流示例](#9-端到端数据流示例)
10. [模块依赖关系图](#10-模块依赖关系图)
11. [关键设计决策总结](#11-关键设计决策总结)

---

## 1. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Engram Memory System                          │
│                                                                        │
│   Source of Truth: ~/.engram/memories/*.md (Markdown + YAML frontmatter)│
│                                                                        │
│   ┌──────────────┐         ┌───────────────┐         ┌──────────────┐ │
│   │  Write Path   │         │  Derived Index │         │  Read Path    │ │
│   │  (remember)   │───────▶│  (rebuildable) │◀────────│  (search)     │ │
│   └──────────────┘         │ ChromaDB+SQLite│         └──────────────┘ │
│                             └───────────────┘                          │
│   ┌──────────────┐         ┌───────────────┐         ┌──────────────┐ │
│   │  Facts Store  │         │  Projects     │         │  LLM Layer   │ │
│   │  facts/*.md   │         │  projects/*.md│         │  (optional)  │ │
│   └──────────────┘         └───────────────┘         └──────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

系统由三条主要数据通路构成：

- **写入路径**（remember / ingest）：将用户输入经过质量门控、去重、分块后写入 Markdown 文件，再更新索引和提取事实。
- **读取路径**（search / recall / wake-up）：通过四层检索栈（L0–L3）从索引中检索相关记忆，可选 LLM 重排。
- **维护路径**（decay / rebuild）：后台衰减不活跃记忆的重要性，或全量重建索引。

---

## 2. 文件系统结构

```
~/.engram/
├── config.yaml              # 全局配置（LLM provider、rerank 设置等）
├── identity.md              # L0 身份文件（给 AI agent 读的 "我是谁"）
├── memories/                # ★ 核心：所有记忆的 Markdown 文件
│   ├── mem_2026-01-15_a3f7c2.md
│   ├── mem_2026-01-16_b8e4d1.md
│   └── ...
├── projects/                # 项目注册表（每个项目一个 .md）
│   ├── saas-app.md
│   ├── engram.md
│   └── _index.md            # 项目索引（自动生成）
├── facts/                   # 结构化知识（每个项目一个 .md）
│   ├── saas-app.md          # 含 ## Current / ## Expired / ## Conflicts
│   └── engram.md
└── .index/                  # ★ 派生索引（可删除重建）
    ├── vectors.chroma/      # ChromaDB 向量数据库（语义搜索）
    └── meta.sqlite3         # SQLite 元数据（结构化查询 + 去重）
```

| 目录/文件 | 角色 | 是否可重建 |
|---|---|---|
| `memories/*.md` | 唯一真相源 — 所有记忆 | ❌ 不可重建（源数据） |
| `projects/*.md` | 项目注册 & 元信息 | ❌ 不可重建（用户创建） |
| `facts/*.md` | 结构化知识三元组 | ❌ 不可重建（提取结果） |
| `identity.md` | AI agent 身份描述 | ❌ 不可重建（用户编辑） |
| `config.yaml` | 系统配置 | ❌ 不可重建（用户配置） |
| `.index/` | ChromaDB + SQLite | ✅ 可 `engram rebuild-index` 重建 |

---

## 3. 写入路径（Write Path）

### 3.1 完整写入流水线 — `remember.py::remember()`

写入操作经过 **7 步流水线**。Step 1–3 是关键路径（失败即中止），Step 4–7 是非关键步骤（失败不影响记忆存储）。

```
用户输入 "remember X"
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Step 1: Quality Gate (quality.py)                  │
│                                                     │
│  纯启发式规则，无 LLM 依赖：                         │
│  ├── 空内容/纯空白 → REJECT                         │
│  ├── 代码文件 prose_ratio < 0.15 → REJECT           │
│  ├── 太短 < 50 字符 → 存，但 importance = 0.5       │
│  ├── 噪音模式 ("ok","sure","Here's...") → REJECT    │
│  ├── 决策标记 ("decided","chose") → importance = 4.0│
│  ├── 里程碑标记 ("launched","shipped") → imp = 4.0  │
│  ├── 问题标记 ("bug","broken") → importance = 3.5   │
│  └── 默认 → importance = 3.0                        │
└───────────────────────┬─────────────────────────────┘
                        │ importance 值
                        ▼
┌─────────────────────────────────────────────────────┐
│  Step 2: Dedup Check (dedup.py)                     │
│                                                     │
│  三级去重策略：                                      │
│  ├── L1: SHA256 哈希精确匹配                         │
│  │   normalize(content) → sha256 → SQLite 查找      │
│  │   命中 → is_duplicate=True, reason="hash_match"  │
│  │                                                   │
│  ├── L2: 向量相似度 ≥ 0.92                           │
│  │   ChromaDB top-3 查询                             │
│  │   similarity ≥ 0.92 → "exact_semantic"           │
│  │                                                   │
│  └── L3: 0.82–0.92 范围                              │
│      同项目 AND 同类型 → "merge_candidate"           │
│      否则 → 不重复，保留                              │
└───────────────────────┬─────────────────────────────┘
                        │ 通过去重
                        ▼
┌─────────────────────────────────────────────────────┐
│  Step 3: Write Markdown (store.py)                  │
│  ★ Source of Truth 写入点                            │
│                                                     │
│  生成 ID: mem_{YYYY-MM-DD}_{sha256[:6]}             │
│  生成文件名: slugify(content) + 冲突后缀 (_1, _2)    │
│                                                     │
│  写入 memories/mem_2026-01-15_a3f7c2.md:            │
│  ┌──────────────────────────────┐                   │
│  │ ---                          │                   │
│  │ id: mem_2026-01-15_a3f7c2   │                   │
│  │ project: saas-app            │                   │
│  │ topics: [auth, migration]    │                   │
│  │ memory_type: note            │                   │
│  │ importance: 4.0              │                   │
│  │ created: 2026-01-15T10:30:00 │                   │
│  │ source: cli                  │                   │
│  │ access_count: 0              │                   │
│  │ ---                          │                   │
│  │                              │                   │
│  │ We switched from MongoDB     │                   │
│  │ to PostgreSQL for the        │                   │
│  │ saas-app project.            │                   │
│  └──────────────────────────────┘                   │
└───────────────────────┬─────────────────────────────┘
                        │ filepath
                        ▼
┌─────────────────────────────────────────────────────┐
│  Step 4: Update Index (index.py) [非关键]            │
│                                                     │
│  ChromaDB:                                          │
│  └── upsert(id, document=body, metadata={...})      │
│      embedding 由 ChromaDB 内置                      │
│      all-MiniLM-L6-v2 自动生成 (384 维)             │
│                                                     │
│  SQLite:                                            │
│  └── INSERT/UPDATE memory_index                      │
│      (id, project, topics, importance,              │
│       content_hash, access_count, ...)              │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  Step 5: Extract Facts (extract.py) [非关键]         │
│                                                     │
│  ┌─ IF LLM 可用 ─────────────────────────────────┐ │
│  │  发送 prompt 到 ollama/openai/anthropic        │ │
│  │  prompt 包含:                                   │ │
│  │   - 已知事实列表 (用于冲突检测)                  │ │
│  │   - 待分析文本                                   │ │
│  │  返回 JSON:                                      │ │
│  │   [{subject, predicate, object,                 │ │
│  │     confidence, temporal, conflicts_with}, ...]  │ │
│  └─────────────────────────────────────────────────┘ │
│                                                     │
│  ┌─ ELSE 启发式提取 (6 类正则) ──────────────────┐  │
│  │  ├── 技术选择: "uses X", "chose X"            │  │
│  │  ├── 转换: "switched from X to Y"             │  │
│  │  ├── 分配: "X will handle Y"                  │  │
│  │  ├── 决策: "decided to X"                     │  │
│  │  ├── 状态: "X is deployed/broken"             │  │
│  │  └── 指标: "DAU: N", "costs $X/mo"            │  │
│  └───────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────┘
                        │ FactCandidate[]
                        ▼
┌─────────────────────────────────────────────────────┐
│  Step 6: Add Facts + Conflict Detection [非关键]     │
│  (facts.py + conflicts.py)                          │
│                                                     │
│  对每个 FactCandidate:                               │
│  ├── 查重: 相同 (S, P, O) → 跳过                    │
│  ├── 冲突检测: 同 S+P 但不同 O (仅排他谓词)          │
│  │   排他谓词: uses_database, uses_auth, status,    │
│  │             assigned_to, led_by, ...             │
│  │   非排他谓词: loves, knows, works_with → 不触发   │
│  │                                                   │
│  └── 冲突分类 (4 类):                                │
│      ├── temporal_succession                         │
│      │   新日期 > 旧日期 → 自动解决 (conf=1.0)      │
│      ├── implicit_supersede                          │
│      │   检测到替换语言 → 自动解决 (conf=0.85)       │
│      ├── opinion_change                              │
│      │   观点/偏好变化 → 自动解决 (两个都保留)       │
│      └── hard_contradiction                          │
│          无法判断 → 标记待用户解决 (both conf=0.5)    │
│                                                     │
│  写入 facts/{project}.md:                            │
│    ## Current  ← 活跃事实                            │
│    ## Expired  ← 已过期事实（保留历史）               │
│    ## Conflicts ← 待解决冲突                          │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  Step 7: Update Project (projects.py) [非关键]       │
│                                                     │
│  更新 projects/{project}.md:                         │
│  └── last_active = now                               │
└─────────────────────────────────────────────────────┘
```

### 3.2 批量写入 — `ingest.py`

用于从文件或目录批量摄入记忆。

```
目录/文件输入
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  ingest.py — 多格式摄入                                          │
│                                                                  │
│  1. 文件类型检测                                                  │
│     ├── .md        → markdown                                    │
│     ├── .py/.js/.ts/.go/.rs → code                               │
│     ├── .txt       → text                                        │
│     ├── .json/.csv → data                                        │
│     └── .jpg/.png  → binary (跳过)                               │
│                                                                  │
│  2. 话题推导 (_derive_topics)                                     │
│     └── 从文件路径提取: src/auth/clerk.py → ["auth", "clerk"]     │
│                                                                  │
│  3. 分块策略 (chunking):                                          │
│     ┌──────────────────┬──────────────────────────────────────┐  │
│     │ 格式              │ 策略                                 │  │
│     ├──────────────────┼──────────────────────────────────────┤  │
│     │ markdown          │ 按 ## 标题分割，检测决策标记          │  │
│     │ text              │ 按双换行(段落)分割，合并过小段落      │  │
│     │ code              │ 按函数/类定义分割                     │  │
│     │                   │ 支持: Python, JS/TS, Go, Rust,       │  │
│     │                   │       Ruby, Java                     │  │
│     │ conversation      │ 按说话人轮次分割                     │  │
│     │                   │ 检测 > 引用和 User:/Assistant: 标签  │  │
│     └──────────────────┴──────────────────────────────────────┘  │
│                                                                  │
│  4. 每个 chunk → remember_batch() → 走完整 7 步流水线             │
│                                                                  │
│  返回 IngestResult:                                               │
│     total_files, total_chunks, memories_created,                 │
│     duplicates_skipped, quality_rejected, errors                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. 读取路径（Read Path）

### 4.1 四层检索栈 — `layers.py::MemoryStack`

检索采用四层栈架构，从静态到动态逐层加深，按需触发。

```
┌─────────────────────────────────────────────────────────────────────┐
│                      4-Layer Retrieval Stack                        │
│                                                                     │
│  L0: Identity (静态, 始终加载)                                       │
│  ├── 读取 ~/.engram/identity.md                                     │
│  ├── 去除 YAML frontmatter，返回 body                               │
│  └── 用���: 注入 AI 系统提示 "你在为谁工作"                           │
│                                                                     │
│  L1: Working Set (自动生成, ≤500 tokens)                             │
│  ├── 项目模式 (project ≠ None):                                     │
│  │   ├── 项目概览 (projects/{project}.md, 前 200 tokens)            │
│  │   ├── 活跃事实 (facts/{project}.md → ## Current)                 │
│  │   ├── 最近 5 条记忆 (SQLite ORDER BY created DESC LIMIT 5)       │
│  │   └── 未解决冲突 (facts/{project}.md → ## Conflicts)             │
│  └── 跨项目模式 (project = None):                                   │
│      ├── 所有活跃项目列表                                           │
│      ├── 全局最近 5 条记忆                                          │
│      └── 跨项目未解决冲突                                           │
│                                                                     │
│  L2: Contextual Recall (自动触发, ≤3 条)                             │
│  ├── 1. 项目解析 (resolve_project):                                 │
│  │   ├── 优先级 1: 显式指定                                         │
│  │   ├── 优先级 2: cwd 路径匹配 associated_paths                    │
│  │   ├── 优先级 3: 消息中的别名/关键词匹配                           │
│  │   └── 优先级 4: None → 跨项目模式                                 │
│  ├── 2. 语义搜索 (vector_search / vector_search_reranked)           │
│  └── 3. 格式化为上下文字符串                                        │
│                                                                     │
│  L3: Deep Search (用户触发, ≤5 条)                                   │
│  ├── 不限项目范围的语义搜索                                          │
│  ├���─ 可选 LLM 重排                                                  │
│  └── 详细输出 (项目、话题、类型、日期、相似度)                       │
│                                                                     │
│  ─── 会话启动 wake_up() ───                                         │
│  └── 返回 L0 + L1 合并文本，注入 AI 系统提示                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 搜索数据流详解 — `search.py::search()`

`search()` 是带增强（事实 + 冲突）的语义搜索入口。

```
search(query, project, topics, n=5)
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 1: 向量搜索                                                │
│                                                                  │
│  ┌─ IF rerank_enabled (LLM 可用且未禁用) ─────────────────────┐ │
│  │                                                             │ │
│  │  Stage A: ChromaDB.query(top-20 candidates)                 │ │
│  │  ├── query text → embedding (all-MiniLM-L6-v2)             │ │
│  │  ├── cosine distance → similarity = 1.0 - distance         │ │
│  │  ├── where 过滤: project, memory_type                      │ │
│  │  └── 后过滤: topics (ANY 匹配)                             │ │
│  │                                                             │ │
│  │  Stage B: LLM Rerank (rerank.py)                            │ │
│  │  ├── 构建 prompt:                                           │ │
│  │  │   - 问题 + 20 个候选文档 (各截取 300 字符)              │ │
│  │  │   - 要求返回 JSON 数组 [3, 1, 7, ...]                   │ │
│  │  ├── 调用 LLM (ollama/openai/anthropic, timeout=15s)       │ │
│  │  ├── 解析响应:                                              │ │
│  │  │   ├── 尝试 1: JSON 数组 [3, 1, 7]                       │ │
│  │  │   ├── 尝试 2: Markdown 包裹 ```json [3,1,7] ```         │ │
│  │  │   └── 尝试 3: 文本中提取数字                              │ │
│  │  ├── 1-based → 0-based 索引映射                              │ │
│  │  ├── 去重 + 范围校验                                         │ │
│  │  └── 失败回退 → 返回原始向量排序                              │ │
│  │                                                             │ │
│  └─ ELSE (纯向量) ──────────────────────────────────────────┘  │
│     └── ChromaDB.query(top-N) → 直接返回                       │
│                                                                  │
│  Step 2: 更新访问统计                                             │
│  └── SQLite: access_count++, last_accessed = now                 │
│                                                                  │
│  Step 3: 事实增强                                                 │
│  ├── 从内容中提取实体名:                                          │
│  │   ├── 大写词 (mid-sentence): "Clerk", "Maya"                 │
│  │   ├── CamelCase: "FastAPI", "TypeScript"                     │
│  │   ├── 全大写缩写: "API", "SQL", "AWS"                        │
│  │   ├── 引号内容: "MongoDB", 'PostgreSQL'                      │
│  │   └── 项目名自身                                              │
│  ├── 对每个实体查找 facts/{project}.md 中的相关事实               │
│  └── 附加到 EnrichedHit.related_facts                             │
│                                                                  │
│  Step 4: 冲突检测                                                 │
│  ├── 读取 facts/{project}.md → ## Conflicts 段                    │
│  ├── 检查冲突实体是否出现在搜索结果内容中                          │
│  └── 附加到 EnrichedHit.conflicts                                 │
│                                                                  │
│  返回 SearchResults:                                              │
│    hits: List[EnrichedHit]                                       │
│    query, project, total_facts_found, unresolved_conflicts       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. 索引层详解

### 5.1 双索引架构

Engram 维护两个派生索引，各司其职：

```
┌──────────────────────────────────┬──────────────────────────────────┐
│         ChromaDB (向量)           │          SQLite (结构化)          │
├──────────────────────────────────┼──────────────────────────────────┤
│ Collection: engram_memories      │ Table: memory_index              │
│ Distance metric: cosine          │                                  │
│ Index algorithm: HNSW            │ Columns:                         │
│                                  │   id TEXT PK                     │
│ Stored per document:             │   project TEXT                   │
│   id: memory_id                  │   topics TEXT (JSON)             │
│   document: body text            │   memory_type TEXT               │
│   embedding: auto-generated      │   importance REAL                │
│     model: all-MiniLM-L6-v2     │   created TEXT                   │
│     dimensions: 384              │   file_path TEXT                 │
│   metadata:                      │   content_hash TEXT              │
│     project (str)                │   access_count INTEGER           │
│     topics (JSON str)            │   last_accessed TEXT             │
│     memory_type (str)            │   indexed_at TEXT                │
│     importance (float)           │                                  │
│     created (str)                │ Indexes:                         │
│     file_path (str)              │   idx_mi_project                 │
│                                  │   idx_mi_created (DESC)          │
│ 用途:                             │   idx_mi_importance (DESC)       │
│   语义搜索                        │   idx_mi_content_hash            │
│   query → embedding → cosine     │                                  │
│   HNSW 近似最近邻                │ 用途:                             │
│                                  │   结构化查询 (按项目/日期/类型)   │
│                                  │   去重 (content_hash 查找)        │
│                                  │   访问统计 (衰减引擎用)           │
│                                  │   增量更新 (indexed_at 比较)      │
├──────────────────────────────────┴──────────────────────────────────┤
│ Table: index_meta                                                   │
│   key="last_rebuild_time" → value=ISO timestamp                     │
│   用途: 增量更新的时间基准                                           │
└─────────────────────────────────────────────────────────��───────────┘
```

### 5.2 索引维护操作

| 操作 | 方法 | 说明 |
|---|---|---|
| 全量重建 | `rebuild()` | 清空两个索引 → 扫描 `memories/*.md` → 逐文件重建 |
| 增量更新 | `incremental_update()` | 比较文件 `mtime > indexed_at` → 只更新变化的文件 + 删除已不存在文件的索引条目 |
| 单文件索引 | `index_memory(filepath)` | 解析 frontmatter → ChromaDB upsert + SQLite upsert |
| 删除索引 | `remove_from_index(id)` | 从两个索引中移除 |

---

## 6. 后台维护（Decay Engine）

`decay.py::run_decay()` 定期运行，调整记忆的重要性分数。

```
扫描 SQLite memory_index 所有条目
    │
    ├── 衰减规则 (按严重程度排列, 互斥):
    │   ┌────────────────────┬──────────────────┐
    │   │ 条件                │ 操作              │
    │   ├────────────────────┼──────────────────┤
    │   │ 180 天未访问        │ importance × 0.85│
    │   │ 90 天未访问         │ importance × 0.90│
    │   │ 30 天未访问         │ importance × 0.95│
    │   └────────────────────┴──────────────────┘
    │
    ├── 提升规则 (一次性, 用 frontmatter 标记防重复):
    │   ┌────────────────────┬──────────────────┐
    │   │ 条件                │ 操作              │
    │   ├────────────────────┼──────────────────┤
    │   │ access_count ≥ 3   │ importance + 1.0 │
    │   │ access_count ≥ 10  │ 额外 + 0.5       │
    │   └────────────────────┴──────────────────┘
    │
    ├── importance 范围: [0.1, 5.0]
    │
    └── 写回:
        ├── Markdown frontmatter (importance 字段)
        └── SQLite memory_index (importance 列)
```

支持 `engram decay --dry-run` 预览效果而不实际修改。

---

## 7. LLM 集成点

Engram 不直接调用 LLM API。LLM 能力通过 `llm_fn` 回调由宿主 agent（如 echo-code）注入。

详见 [llm-integration.md](./llm-integration.md)。

### 7.1 集成点一览

| 集成点 | 模块 | 调用时机 | 失败行为 |
|---|---|---|---|
| **事实提取** | `extract.py` → `llm.py` | `remember()` 写入时 | 回退到启发式正则提取 |
| **搜索重排** | `rerank.py` → `llm.py` | `search()` 读取时 | 回退到原始向量排序 |
| **查询重写** | `llm.py` | `search()` 读取时 | 使用原始查询 |
| **时间推理** | `llm.py` | `search()` 读取时 | 跳过 |

### 7.2 配置

```yaml
# ~/.engram/config.yaml
llm:
  rerank: true                # 启用 LLM 重排 (默认: true)
  rerank_candidates: 20       # 重排候选数量 (默认: 20)
  query_rewrite: false         # 查询重写 (默认: false, 增加延迟)
  temporal_reasoning: true     # 时间推理 (默认: true)
```

环境变量覆盖（优先级高于配置文件）：

| 环境变量 | 对应配置 |
|---|---|
| `ENGRAM_RERANK` | `llm.rerank` (设为 `0`/`false` 禁用) |
| `ENGRAM_RERANK_CANDIDATES` | `llm.rerank_candidates` |
| `ENGRAM_QUERY_REWRITE` | `llm.query_rewrite` |
| `ENGRAM_TEMPORAL_REASONING` | `llm.temporal_reasoning` |

---

## 8. CLI 入口

| 命令 | 调用链 | 说明 |
|---|---|---|
| `engram init` | `config.py::EngramConfig.init()` | 初始化 `~/.engram/` 目录结构 |
| `engram remember <text>` | `remember.py::remember()` | 记住内容（7 步流水线） |
| `engram search <query>` | `layers.py::MemoryStack.search()` | L3 深度搜索 |
| `engram wake-up` | `layers.py::MemoryStack.wake_up()` | L0+L1 会话启动上下文 |
| `engram recall <message>` | `layers.py::MemoryStack.recall()` | L2 上下文召回 |
| `engram project create` | `projects.py::create_project()` | 创建项目 |
| `engram project list` | `projects.py::list_projects()` | 列出项目 |
| `engram project archive` | `projects.py::archive_project()` | 归档项目 |
| `engram facts <project>` | `facts.py::get_active_facts()` | 查看项目事实 |
| `engram conflicts` | `facts.py::get_unresolved_conflicts()` | 查看未解决冲突 |
| `engram rebuild-index` | `index.py::IndexManager.rebuild()` | 全量重建索引 |
| `engram decay` | `decay.py::run_decay()` | 运行衰减引擎 |
| `engram status` | `layers.py::MemoryStack.get_status()` | 系统状态 |

---

## 9. 端到端数据流示例

### 场景：用户记录 "We switched from MongoDB to PostgreSQL for the saas-app project"

**写入阶段：**

```
1. CLI: engram remember "We switched from MongoDB to PostgreSQL" --project saas-app

2. Quality Gate (quality.py):
   ✓ 长度 > 50 字符
   ✓ 不是噪音模式
   ★ 检测到决策标记 "switched" → importance = 4.0

3. Dedup (dedup.py):
   L1: SHA256 hash → SQLite 查找 → 未命中
   L2: 向量搜索 top-3 → 最高相似度 0.75 → 不重复

4. Write Markdown (store.py):
   → memories/mem_2026-06-15_c7a3b1.md
   ---
   id: mem_2026-06-15_c7a3b1
   project: saas-app
   importance: 4.0
   memory_type: note
   created: 2026-06-15T10:30:00
   ---
   We switched from MongoDB to PostgreSQL

5. Update Index (index.py):
   ChromaDB: upsert → all-MiniLM-L6-v2 编码为 384 维向量
   SQLite: INSERT INTO memory_index

6. Extract Facts (extract.py, 启发式):
   匹配 "switched from X to Y" 模式
   → FactCandidate(
       subject="saas-app", predicate="uses",
       object="PostgreSQL",
       conflicts_with="saas-app → uses → MongoDB"
     )

7. Add Fact + Conflict (facts.py + conflicts.py):
   查找 facts/saas-app.md → Current 段
   发现: saas-app → uses_database → MongoDB
   冲突分类: implicit_supersede (检测到 "switched from")
   自动解决:
     MongoDB 事实 → ## Expired
     PostgreSQL 事实 → ## Current

8. Update Project (projects.py):
   projects/saas-app.md → last_active = 2026-06-15T10:30:00
```

**读取阶段：**

```
9. CLI: engram search "what database does saas-app use"

10. search() Step 1 — 向量搜索 + 重排:
    Stage A: ChromaDB.query("what database does saas-app use", top-20)
      候选包含 "switched from MongoDB to PostgreSQL" (向量排名第 3)

    Stage B: LLM Rerank (rerank.py)
      LLM 判断: "这条直接回答了数据库问题"
      → 提升到排名第 1

11. search() Step 3 — 事实增强:
    提取实体: "saas-app", "PostgreSQL", "MongoDB"
    查找 facts/saas-app.md:
      ✓ saas-app → uses_database → PostgreSQL (active)
      ✓ saas-app → uses_database → MongoDB (expired)
    附加到结果

12. 最终输出:
    🔍 Search: "what database does saas-app use" (project: saas-app)

    1. [0.94] We switched from MongoDB to PostgreSQL
       📌 Facts: saas-app→uses_database→PostgreSQL (since 2026-06)

    📊 2 related fact(s) found
```

---

## 10. 模块依赖关系图

```
                        config.py
                      ╱     │     ╲
                    ╱       │       ╲
              store.py    index.py    projects.py
                 │       ╱    ╲          │
                 │     ╱       ╲         │
              dedup.py     search.py     │
                 │            │          │
                 ▼            ▼          │
            quality.py     facts.py  ◄───┘
                 │            │
                 │       conflicts.py
                 │            │
                 ▼            ▼
             remember.py  ◄───┘
                 │
                 ▼
             ingest.py
                 │
                 ▼
             layers.py  (MemoryStack — 统一读取接口)
                 │
                 ▼
              cli.py    (命令行入口)

     独立模块:
       rerank.py   ← 由 index.py / search.py / layers.py 调用
       extract.py  ← 由 remember.py 调用
       decay.py    ← 由 cli.py 直接调用
```

### 模块职责速查表

| 模块 | 行数 | 职责 |
|---|---|---|
| `config.py` | ~280 | 配置管理、目录结构、LLM/rerank 设置 |
| `store.py` | ~420 | Markdown 文件读写（Source of Truth） |
| `index.py` | ~500 | 双索引管理（ChromaDB + SQLite） |
| `search.py` | ~250 | 带增强的语义搜索（事实 + 冲突） |
| `layers.py` | ~400 | 四层检索栈（L0–L3）统一接口 |
| `remember.py` | ~200 | 7 步写入流水线 |
| `ingest.py` | ~990 | 多格式批量摄入 + 分块 |
| `extract.py` | ~550 | 事实提取（LLM + 启发式双模式） |
| `facts.py` | ~500 | 事实文件管理（CRUD + 查询） |
| `conflicts.py` | ~350 | 冲突分类 + 解决引擎（4 类冲突） |
| `projects.py` | ~300 | 项目注册 + 上下文路由 |
| `dedup.py` | ~150 | 三级语义去重 |
| `quality.py` | ~200 | 质量门控（启发式规则） |
| `decay.py` | ~200 | 衰减 + 提升引擎 |
| `rerank.py` | ~300 | LLM 重排（listwise） |
| `cli.py` | ~580 | 命令行入口 |

---

## 11. 关键设计决策总结

| 决策 | 选择 | 原因 |
|---|---|---|
| Source of Truth | Markdown 文件 | 人类可读、可 Git 版本控制、不依赖任何数据库 |
| 向量引擎 | ChromaDB (内置 all-MiniLM-L6-v2) | 零配置，自动 embedding，384 维 |
| 元数据存储 | SQLite | 结构化查询、去重 hash 查找、访问计数 |
| LLM 调用方式 | llm_fn 回调协议 | 宿主 agent 注入，零 SDK 依赖 |
| 事实提取 | 启发式 + LLM 双模式 | 无 LLM 也能工作，有 LLM 更准 |
| 冲突检测 | 4 类分类 + 自动解决 | 仅排他谓词触发冲突，大部分可自动解决 |
| 去重策略 | 3 级 (hash→向量→合并判断) | 从精确到模糊，逐级放宽 |
| 搜索重排 | Listwise LLM rerank | 1 次 LLM 调用处理 20 候选，成本低 |
| 检索架构 | 4 层栈 (L0–L3) | 从静态到动态逐层加深，按需触发 |
| 索引策略 | 全量重建 + 增量更新 | 索引是派生物，随时可重建 |
| 重要性管理 | 自动衰减 + 访问提升 | 模拟人类遗忘曲线，常用记忆自然浮升 |
