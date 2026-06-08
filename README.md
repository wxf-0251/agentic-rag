# Agentic RAG — 自主型深度研究 Agent

> 基于多智能体推理循环、并行检索和流式 UI 的 Agentic RAG 系统，自动生成 **20 页带引用的深度研究报告**。

## 概述

传统的 RAG（检索增强生成）是"一次检索 → 一次生成"的线性流程，数据覆盖往往不够全面。本系统采用**多智能体推理循环**：拆解问题 → 并行检索 → 撰写章节 → 质量审核 → 不合格则重新检索，循环直到每个章节达到发表标准。

**输入示例：** "分析 2026 年人形机器人的核心供应链及投资风险"

**输出结果：** 一份 20 页的研究报告，包含行内引用标注、自动生成的图表，支持导出 PDF / Markdown 格式。

## 架构

```
用户查询
    │
    ▼
┌───────────────────────────────┐
│ 阶段1：规划师 Agent（Opus 4）   │
│  · 将查询拆解为 5-7 个子问题    │
│  · 生成报告大纲               │
│  · 按数据来源类型归类子问题     │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ 阶段2：研究员 Agent            │
│       （Haiku，N 个并行）      │
│  · 按子问题类型路由数据源：     │
│    · 最新资讯 → Tavily 网页搜索 │
│    · 技术概念 → Chroma 本地知识库│
│    · 财务数据 → Yahoo Finance  │
│    · 混合类型 → Tavily + Chroma│
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ 阶段3：撰稿人 Agent（Opus 4）  │
│  · 合成单个章节                │
│  · 输出 Markdown + <chart> 标签│
│    + 行内引用                  │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│ 阶段4：审核人 Agent（Opus 4）  │
│  · 引用覆盖率检查              │
│  · 数据一致性校验              │
│  · 逻辑漏洞检测                │
│  → 通过：进入下一章节           │
│  → 不通过：重新检索（最多3次）  │
└───────────────────────────────┘
```

## 技术栈

| 层级 | 选型 | 理由 |
|------|------|------|
| 大模型（规划/撰稿/审核） | Claude Opus 4（Anthropic API） | 工具调用能力强，200K 长上下文适合长报告 |
| 大模型（研究员） | Claude Haiku 4 | 成本仅为 Opus 的 1/10，工具调用能力相同 |
| Agent 编排 | LangGraph（StateGraph） | 原生支持条件边实现推理循环，内置检查点/流式输出 |
| 网页搜索 | Tavily Search API | AI 优化的结构化搜索结果，$0.003/次 |
| 向量数据库 | ChromaDB（本地持久化） | 存储领域报告/书籍，Tavily 不可用时的备选 |
| 金融数据 | yfinance（Yahoo Finance） | 免费，无需 API Key |
| 嵌入模型 | OpenAI text-embedding-3-small | 性价比高，1536 维 |
| 前端界面 | Streamlit | 快速构建 UI，原生支持流式输出 |
| 图表生成 | matplotlib | 从撰稿人输出的 `<chart>` 标签自动生成 |
| PDF 导出 | weasyprint | Markdown → PDF 转换 |

## 项目结构

```
agentic_rag/
├── main.py                    # Streamlit 入口
├── config.py                  # API 密钥、模型配置、环境变量
├── cli.py                     # 命令行入口
├── requirements.txt           # 依赖清单
├── .env.example               # 环境变量模板
├── graph/
│   ├── __init__.py
│   ├── state.py               # AgentState TypedDict + 数据模型定义
│   └── workflow.py            # LangGraph StateGraph 工作流定义
├── agents/
│   ├── __init__.py
│   ├── planner.py             # 规划师：问题拆解 + 大纲生成
│   ├── researcher.py          # 研究员：并行检索 + 工具路由
│   ├── writer.py              # 撰稿人：章节合成 + 引用强制
│   └── reviewer.py            # 审核人：质量关卡（引用/一致性/逻辑）
├── tools/
│   ├── __init__.py
│   ├── search.py              # Tavily 网页搜索封装
│   ├── chroma_store.py        # ChromaDB 向量检索
│   ├── finance.py             # Yahoo Finance 金融数据
│   ├── source_quality.py      # 信源可信度评分
│   └── fact_verifier.py       # 跨源事实核查
├── prompts/
│   ├── planner.md             # 规划师系统提示词
│   ├── researcher.md          # 研究员系统提示词
│   ├── writer.md              # 撰稿人系统提示词
│   └── reviewer.md            # 审核人系统提示词
├── ui/
│   ├── __init__.py
│   ├── components.py          # Streamlit 可复用组件
│   ├── renderer.py            # Markdown + <chart> → matplotlib 图表
│   ├── report_sections.py     # 报告章节渲染
│   └── stream_events.py       # 流式事件类型定义
└── tests/
    ├── test_state.py          # 状态模型测试
    ├── test_search.py         # Tavily 搜索测试
    ├── test_chroma_store.py   # ChromaDB 测试
    ├── test_finance.py        # 金融数据测试
    ├── test_planner.py        # 规划师测试
    ├── test_researcher.py     # 研究员测试
    ├── test_writer.py         # 撰稿人测试
    ├── test_reviewer.py       # 审核人测试
    ├── test_workflow.py       # 工作流测试
    ├── test_renderer.py       # 渲染器测试
    ├── test_integration.py    # 端到端集成测试
    ├── test_config.py
    ├── test_source_quality.py
    ├── test_fact_verifier.py
    ├── test_report_sections.py
    ├── test_stream_events.py
    └── test_text_quality.py
```

## 快速开始

### 前置条件

- Python 3.11+
- Anthropic API Key（[console.anthropic.com](https://console.anthropic.com) 获取）
- Tavily API Key（[tavily.com](https://tavily.com) 获取）
- OpenAI API Key（用于 ChromaDB 嵌入，[platform.openai.com](https://platform.openai.com) 获取）

### 安装

```bash
# 克隆仓库
git clone https://github.com/wxf-0251/agentic-rag.git
cd agentic-rag

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 API 密钥
```

### 启动应用

**Streamlit 界面（推荐）：**
```bash
streamlit run agentic_rag/main.py
```

**命令行模式：**
```bash
python -m agentic_rag.cli "分析 2026 年人形机器人核心供应链及投资风险"
```

### 运行测试

```bash
pytest tests/ -v
```

## 核心特性

- **多智能体推理循环** — 规划师 → 研究员（并行）→ 撰稿人 → 审核人 →（不合格则重新检索）
- **智能工具路由** — 根据子问题类型自动选择 Tavily / ChromaDB / Yahoo Finance 数据源
- **实时流式输出** — 在 Streamlit 界面中实时看到研究进度、草稿章节和引用来源
- **行内引用标注** — 每个事实性陈述都有可追溯的来源引用
- **自动图表生成** — 撰稿人输出 `<chart>` 标签，渲染器自动转为 matplotlib 图表
- **质量关卡** — 审核人检查引用覆盖率、数据一致性和逻辑漏洞
- **优雅降级** — 每个外部依赖都有备选方案（如 Tavily 不可用时自动切换 Chroma）
- **成本控制** — 可配置单次报告预算上限，高成本模型启用 prompt 缓存
- **PDF 导出** — 通过 weasyprint 将 Markdown 转为 PDF 格式

## 配置说明

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `ANTHROPIC_API_KEY` |（必填）| Anthropic Claude API 密钥 |
| `TAVILY_API_KEY` |（必填）| Tavily 网页搜索 API 密钥 |
| `OPENAI_API_KEY` |（必填）| OpenAI API 密钥（用于文本嵌入） |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | ChromaDB 本地持久化目录 |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | 嵌入模型名称 |
| `MAX_RESEARCH_ITERATIONS` | `3` | 单章节最大重新检索次数 |
| `MAX_BUDGET_USD` | `2.00` | 单次报告硬性费用上限（美元） |
| `PLANNER_MODEL` | `claude-opus-4-20250514` | 规划师使用的模型 |
| `WRITER_MODEL` | `claude-opus-4-20250514` | 撰稿人使用的模型 |
| `REVIEWER_MODEL` | `claude-opus-4-20250514` | 审核人使用的模型 |
| `RESEARCHER_MODEL` | `claude-haiku-4-20250514` | 研究员使用的模型 |

## 设计原则

1. **非简单线性流水线** — 审核人可以触发重新检索，形成推理闭环
2. **按子问题类型路由数据源** — 不同问题走不同检索通道，而非一刀切
3. **LangGraph StateGraph 共享类型化状态** — 所有 Agent 读写同一份 TypedDict
4. **全阶段流式输出** — 用户看到实时进度，而非对着转圈圈干等 3 分钟
5. **优雅降级** — 每个外部依赖都有备选路径，单点故障不影响整体流程
6. **Prompt 缓存** — 规划师/撰稿人/审核人的系统提示词启用缓存，成本降低约 90%

## 费用估算

- 每份完整报告（20 页、5-7 章节）约 **$0.50 – $1.50**（美元）
- 硬性费用上限：$2.00 / 报告（可配置）

## 开源协议

MIT

## 致谢

本项目基于以下开源项目构建：

- [Anthropic Claude API](https://docs.anthropic.com)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Tavily Search API](https://tavily.com)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io)
