# Agentic RAG — 自主型深度研究 Agent

> 基于 LangGraph 多智能体协作框架，输入宏观研究问题，自动产出带详细引用的深度研究报告。

## 项目描述

传统 RAG（检索增强生成）存在**检索浅层、易产生幻觉**的痛点——单次检索难以覆盖复杂问题的多个维度，生成内容缺乏可追溯的事实依据。

本系统引入 **任务拆解 → 多源路由检索 → 撰写 → 自我批判**的完整闭环，构建了一个端到端的深度研究报告自动生成系统。用户只需输入一个宏观研究问题，系统自动完成子问题拆解、多源并行检索、章节撰写、事实核查与自我修正，最终输出一份带有行内引用和可追溯来源的完整研究报告。

**输入示例：**"分析 2026 年人形机器人的核心供应链及投资风险"

**输出结果：** 一份带有行内引用标注、自动生成图表的深度研究报告，支持导出 Markdown / PDF 格式。

## 技术栈

| 层级 | 选型 |
|------|------|
| 编程语言 | Python |
| Agent 编排 | LangGraph（StateGraph） |
| 大模型 | Anthropic Claude API（Opus 4.7 / Haiku 4.5） |
| 向量数据库 | ChromaDB |
| 网页搜索 | Tavily Search API |
| 金融数据 | Yahoo Finance（yfinance） |
| 前端界面 | Streamlit |

## 核心工作

### 1. 多智能体协作流水线

设计 **Planner（任务拆解）→ Researcher（多源检索）→ Writer（章节撰写）→ Reviewer（自我批判）** 四角色 LangGraph StateGraph 编排架构：

```
用户查询
    │
    ▼
┌─────────────────────────────────┐
│  Planner（Claude Opus 4.7）      │
│  将宏观问题拆解为 5-7 个子问题    │
│  生成报告大纲，按数据源类型归类   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Researcher（Claude Haiku 4.5）  │
│  N 个并行 — Send API fan-out    │
│  按子问题类型动态路由数据源       │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Writer（Claude Opus 4.7）       │
│  合成单个章节，强制行内引用       │
│  输出 Markdown + <chart> 标签    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Reviewer（Claude Opus 4.7）     │
│  数值交叉校验 + 来源时效性检测    │
│  PASS → 下一章节                 │
│  FAIL → 生成改进关键词，重新检索  │
└─────────────────────────────────┘
```

基于 LangGraph 的 **Send API** 实现子问题并行 fan-out 检索，N 个子问题同时执行而非串行排队，显著降低端到端延迟。

### 2. 多源动态路由

根据子问题类型自动决策调用不同数据源，并对检索结果进行权威度排序：

| 子问题类型 | 路由目标 | 示例 |
|-----------|---------|------|
| `latest_news` | Tavily Search API | 最新融资动态、政策变化 |
| `technical_concept` | Chroma 本地向量数据库 | 谐波减速器原理、力矩传感 |
| `financial_data` | Yahoo Finance | 上市公司股价、市值、财报 |
| `mixed` | Tavily + Chroma 并行 | 综合型问题 |

检索结果基于**域名权威度评分**排序：政府/交易所 > 财经媒体 > 行业报告 > 社交媒体，同时结合时效性加权，确保优先使用高质量信息源。

### 3. 自我批判闭环

Reviewer Agent 对每个章节执行严格的质量审查，不通过则自动触发重新检索：

- **数值-引用交叉匹配**：校验文中数值声明是否能在对应引用的 source snippet 中找到依据
- **来源时效性检测**：识别"当前""最新""近期"等时效性断言，检查引用来源的发布时间是否匹配
- **逻辑一致性检查**：验证结论是否从证据中合理推导，是否存在逻辑跳跃

审查不通过时，Reviewer 自动生成改进搜索关键词，触发 Researcher 重新检索，**最多迭代 3 轮**。3 轮后仍未通过，则强制接受并在报告中标注"该部分数据覆盖有限"，确保报告的事实可靠性。

### 4. 数值事实核查

独立的事实核查模块（`fact_verifier.py`），通过正则表达式提取文中所有数值声明，逐一检查是否在对应引用的 source snippet 中出现：

- 标记**"无出处数据"**：数值声明在全部引用中均无法找到
- 标记**"引用不匹配"**：数值与引用中的原始数据不一致

核查结果反馈给 Reviewer 作为 PASS/FAIL 判断的重要依据。

### 5. 流式交互 UI

基于 Streamlit 构建实时进度面板，可视化展示全流程状态：

- Planner 拆解的子问题列表及数据源归类
- Researcher 各子问题检索结果数及引用来源
- Writer 已完成章节的流式渲染（token 级别实时输出）
- Reviewer 审核结论（✅ PASS / 🔄 FAIL）及 critique 详情
- 右侧面板实时汇总引用来源及链接

### 6. 成本分层调度

按任务复杂度分层选用模型，在保证分析质量的同时控制成本：

| Agent | 模型 | 原因 |
|-------|------|------|
| Planner | Claude Opus 4.7 | 需要强推理能力拆解复杂问题 |
| Writer | Claude Opus 4.7 | 长文撰写需高质量逻辑与表达 |
| Reviewer | Claude Opus 4.7 | 批判性思维需强判断力 |
| Researcher | Claude Haiku 4.5 | 信息提取与汇总，成本为 Opus 的 1/10 |

单次完整研究报告（20 页、5-7 章节）预算约 **$1**，硬性上限 $2.00。Planner/Writer/Reviewer 的系统提示词启用 Anthropic prompt caching，缓存命中后成本降低约 90%。

## 项目结构

```
agentic_rag/
├── main.py                    # Streamlit 入口
├── config.py                  # API 密钥、模型配置、环境变量
├── cli.py                     # 命令行入口
├── requirements.txt           # 依赖清单
├── .env.example               # 环境变量模板
├── graph/
│   ├── state.py               # AgentState TypedDict + 数据模型定义
│   └── workflow.py            # LangGraph StateGraph 工作流（Send API fan-out）
├── agents/
│   ├── planner.py             # Planner：问题拆解 + 大纲生成
│   ├── researcher.py          # Researcher：多源路由检索 + 并行 fan-out
│   ├── writer.py              # Writer：章节撰写 + 引用强制
│   └── reviewer.py            # Reviewer：数值校验 + 时效性检测 + 批判闭环
├── tools/
│   ├── search.py              # Tavily Search API 封装
│   ├── chroma_store.py        # ChromaDB 向量检索
│   ├── finance.py             # Yahoo Finance 金融数据
│   ├── source_quality.py      # 域名权威度评分 + 时效性加权
│   └── fact_verifier.py       # 数值事实核查（正则提取 + 交叉匹配）
├── prompts/
│   ├── planner.md             # 规划师系统提示词
│   ├── researcher.md          # 研究员系统提示词
│   ├── writer.md              # 撰稿人系统提示词
│   └── reviewer.md            # 审核人系统提示词
├── ui/
│   ├── components.py          # Streamlit 可复用组件
│   ├── renderer.py            # Markdown + <chart> → matplotlib 图表
│   ├── report_sections.py     # 报告章节渲染
│   └── stream_events.py       # 流式事件类型定义
└── tests/
    ├── test_integration.py    # 端到端集成测试
    ├── test_workflow.py       # 工作流及 Send API fan-out 测试
    ├── test_planner.py        # 规划师测试
    ├── test_researcher.py     # 研究员 + 路由测试
    ├── test_writer.py         # 撰稿人测试
    ├── test_reviewer.py       # 审核人 + 批判闭环测试
    ├── test_fact_verifier.py  # 数值核查测试
    ├── test_source_quality.py # 信源评分测试
    └── ...                    # 更多测试文件
```

## 快速开始

### 前置条件

- Python 3.11+
- Anthropic API Key（[console.anthropic.com](https://console.anthropic.com)）
- Tavily API Key（[tavily.com](https://tavily.com)）
- OpenAI API Key（用于 ChromaDB 嵌入，[platform.openai.com](https://platform.openai.com)）

### 安装

```bash
git clone https://github.com/wxf-0251/agentic-rag.git
cd agentic-rag

python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 文件，填入 API 密钥
```

### 启动

**Streamlit 界面（推荐）：**
```bash
streamlit run agentic_rag/main.py
```

**命令行模式：**
```bash
python -m agentic_rag.cli "分析 2026 年人形机器人核心供应链及投资风险"
```

### 测试

```bash
pytest tests/ -v
```

## 配置说明

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `ANTHROPIC_API_KEY` |（必填）| Anthropic Claude API 密钥 |
| `TAVILY_API_KEY` |（必填）| Tavily 网页搜索 API 密钥 |
| `OPENAI_API_KEY` |（必填）| OpenAI API 密钥（嵌入用） |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | ChromaDB 本地持久化目录 |
| `MAX_RESEARCH_ITERATIONS` | `3` | 单章节最大重新检索次数 |
| `MAX_BUDGET_USD` | `2.00` | 单次报告硬性费用上限 |
| `PLANNER_MODEL` | `claude-opus-4-20250514` | 规划师模型 |
| `WRITER_MODEL` | `claude-opus-4-20250514` | 撰稿人模型 |
| `REVIEWER_MODEL` | `claude-opus-4-20250514` | 审核人模型 |
| `RESEARCHER_MODEL` | `claude-haiku-4-20250514` | 研究员模型 |

## 开源协议

MIT
