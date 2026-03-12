<div align="center">
    <img src="assets/cover_pic.jpg" height="300px">
</div>

<h1 align="center">GraphAgent：面向图结构的智能语言助手</h1>

<p align="center">
📖 <a href="https://arxiv.org/abs/2412.17029" target="_blank">Arxiv 论文</a> ·
🤗 <a href="https://huggingface.co/GraphAgent/GraphAgent-7B" target="_blank">GraphAgent 模型</a> ·
🤗 <a href="https://huggingface.co/GraphAgent/GraphTokenizer" target="_blank">图分词器模型</a> ·
🤗 <a href="https://huggingface.co/datasets/GraphAgent/GraphAgent-Datasets" target="_blank">GraphAgent 数据集</a>
</p>

> 📘 **English version:** [README.md](README.md)

---

## 目录

1. [项目概览](#-项目概览)
2. [整体架构](#-整体架构)
3. [语义知识图谱（SKG）的自动构建](#-语义知识图谱skg的自动构建)
   - [第一步：脚手架节点提取（Scaffold Node Extraction）](#第一步脚手架节点提取scaffold-node-extraction)
   - [第二步：知识增强文本解析（Knowledge Augmentation / Text Parsing）](#第二步知识增强文本解析knowledge-augmentation--text-parsing)
   - [第三步：关键词提取与图接地（Keyword Extraction & Graph Grounding）](#第三步关键词提取与图接地keyword-extraction--graph-grounding)
4. [迭代式两阶段图生成工作流](#-迭代式两阶段图生成工作流)
   - [阶段一：骨架构建（Skeleton Construction）](#阶段一骨架构建skeleton-construction)
   - [阶段二：知识增强（Knowledge Augmentation）](#阶段二知识增强knowledge-augmentation)
   - [派生关系追踪与边的构建](#派生关系追踪与边的构建)
5. [核心数据结构与代码对应关系](#-核心数据结构与代码对应关系)
6. [快速开始](#-快速开始)
7. [引用](#-引用)

---

## 🌟 项目概览

现实世界的数据同时以**结构化**（如图连接）和**非结构化**（如文本、视觉信息）两种形式存在，包含显式链接（如社交关系、用户行为）和语义实体之间的隐式相互依赖（通常以知识图谱的形式体现）。

**GraphAgent** 是一个自动化的智能体流水线，能够同时处理显式图依赖和隐式图增强语义依赖，适用于：
- **预测任务**：如节点分类（Node Classification）
- **生成任务**：如文本生成（Text Generation）

GraphAgent 由三个核心组件协同工作：

| 组件 | 英文名 | 职责 |
|------|--------|------|
| **图生成智能体** | Graph Generator Agent | 从非结构化文本中自动构建**语义知识图谱（SKG）** |
| **任务规划智能体** | Task Planning Agent | 解析用户指令，制定任务执行计划 |
| **任务执行智能体** | Task Execution Agent | 执行规划好的任务，自动匹配和调用工具 |

---

## 🏗 整体架构

```
用户输入（自然语言指令）
        │
        ▼
┌─────────────────────────────┐
│     任务规划智能体            │  ← task_planning_agent/agent.py
│  解析指令，提取 knowledge_text │
│  和 user_annotation          │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│              图生成智能体（Graph Generator Agent）    │  ← graph_generation_agent/agent.py
│                                                     │
│  阶段一：脚手架节点提取（x_sys_sk_0 / x_sys_sk_1）   │
│  阶段二：知识增强（x_sys_ka）                        │
│  图接地：构建异构图（HeteroData）                    │
└────────────┬────────────────────────────────────────┘
             │ PyG HeteroData（含 edge_index）
             ▼
┌─────────────────────────────┐
│       图分词器                │  ← graph_tokenizer/graph_tokenizer.py
│  将图结构转为连续 Token 序列   │
└────────────┬────────────────┘
             │ 图 Token + 用户指令
             ▼
┌─────────────────────────────┐
│     任务执行智能体            │  ← graph_action_agent/agent.py
│  GraphAgent-8B 多模态 LLM    │
│  完成分类 / 生成等下游任务     │
└─────────────────────────────┘
```

---

## 🔬 语义知识图谱（SKG）的自动构建

语义知识图谱的自动构建是 GraphAgent 的核心创新之一。整个构建过程位于
`GraphAgent-inference/graph_generation_agent/` 目录，通过 **LangChain** 流水线
将多个 LLM 调用串联起来，将非结构化文本转化为结构化异构图。

### 第一步：脚手架节点提取（Scaffold Node Extraction）

**对应文件：**
- `graph_generation_agent/chains.py` → `create_scaffold_node_extraction_chain()`
- `graph_generation_agent/scaffold_node_extraction_prompts.py`（少样本示例）

**系统提示词（System Prompt）核心内容：**

给定包含 `knowledge_text`（知识文本）和 `user_annotation`（用户标注/任务说明）的输入，
LLM 被要求提取一组**高度抽象的顶层概念节点**，称为"脚手架节点"（Scaffold Nodes）。

脚手架节点的特征：
- **高度抽象**：代表文本中的顶层概念或实体（如论文分类任务中的"研究背景"、"方法论"、"关键结果"）
- **无冗余**：每个节点对应文本的一个独立方面
- **任务相关**：根据 `user_annotation` 中的任务说明动态调整

**LLM 输出格式（JSON）：**
```json
{
    "thinking": "分步推理过程...",
    "scaffold_nodes": [
        {"id": 0, "type": "paper", "name": "论文标题"},
        {"id": 1, "type": "research_background", "name": "research_background"},
        {"id": 2, "type": "methodology", "name": "methodology"}
    ]
}
```

---

### 第二步：知识增强文本解析（Knowledge Augmentation / Text Parsing）

**对应文件：**
- `graph_generation_agent/chains.py` → `create_scaffold_text_parsing_chain()`
- `graph_generation_agent/scaffold_text_and_keywords_prompts.py`（少样本示例）

将上一步得到的脚手架节点列表与原始知识文本一同送入 LLM，要求模型为**每个脚手架节点**
从原文中解析出对应的详细文本片段。

**LLM 输出格式（JSON）：**
```json
{
    "scaffold_texts": [
        {"node_id": 0, "text": "该节点对应的详细文本..."},
        {"node_id": 1, "text": "研究背景相关文本..."}
    ]
}
```

---

### 第三步：关键词提取与图接地（Keyword Extraction & Graph Grounding）

**对应文件：**
- `graph_generation_agent/chains.py` → `create_keywords_from_scaffold_text_chain()`
- `graph_generation_agent/pyg_utils.py` → `build_hetero_graph_from_scaffold_keywords_v2()`

对每个脚手架节点对应的文本片段，LLM 进一步提取**关键词节点**及其描述，
然后由 `build_hetero_graph_from_scaffold_keywords_v2()` 将所有节点和边组装成
PyTorch Geometric 的 `HeteroData` 对象。

**图结构特点：**
- **节点类型**：脚手架节点（按 `type` 分类）+ `keyword` 节点
- **边类型**：`(scaffold_type, "has_keyword", "keyword")`
- **边索引（edge_index）**：形状为 `(2, num_edges)` 的张量

---

## 🔄 迭代式两阶段图生成工作流

除了上述三阶段流水线，GraphAgent 还实现了论文中描述的**迭代式两阶段图生成工作流**，
能够以递归方式对图进行细化。

**对应文件：**
- `graph_generation_agent/agent.py` → `run_iterative_two_phase_graph_generation()`
- `graph_generation_agent/agent.py` → `iterative_agent`（LangChain 流水线）
- `graph_generation_agent/pyg_utils.py` → `build_graph_with_derivation_edges()`

### 阶段一：骨架构建（Skeleton Construction）

骨架构建分为**多个迭代步骤**，每个步骤使用不同的系统提示词：

#### 步骤 0（使用 `x_sys_sk_0`）—— 初始顶层节点提取

```
输入：knowledge_text + user_annotation
使用提示词：x_sys_sk_0
输出：顶层脚手架节点列表（根节点，parent_unified_idx = None）
```

`x_sys_sk_0` 指示 LLM 从**完整知识文本**中提取最高层次的抽象概念节点。

#### 步骤 1+（使用 `x_sys_sk_1`）—— 子节点派生

```
输入：父节点的 name + description
使用提示词：x_sys_sk_1
输出：从父节点描述文本中派生出的子节点列表
```

`x_sys_sk_1` 指示 LLM 从**父节点的描述文本**中提炼出更具体、更细粒度的子节点，
这些子节点与父节点之间形成**派生关系（derivation relation）**，直接对应论文中的描述：

> *"if a new node is generated from the textual description of a node in the previous
> iteration, we connect these two nodes."*
>（若一个新节点是由前一迭代中某节点的文本描述生成的，则将这两个节点连接起来。）

---

### 阶段二：知识增强（Knowledge Augmentation）

骨架构建完成后，对**所有节点**（包括各迭代步骤中产生的根节点和子节点）调用
`knowledge_augmentation_chain`（使用提示词 `x_sys_ka`），为每个节点补充：

- `augmented_description`：更丰富的节点描述（结合背景知识）
- `attributes`：节点的关键属性列表

---

### 派生关系追踪与边的构建

#### 数据结构：`derivation_edges`

在迭代过程中，父子关系通过以下字典实时记录：

```python
derivation_edges: dict[int, list[int]]
# 键（key）  = 父节点的 unified_idx（全局统一索引）
# 值（value）= 子节点的 unified_idx 列表
```

例如，若根节点 0 在步骤 1 中派生出子节点 2 和 3，根节点 1 派生出子节点 4，则：

```python
derivation_edges = {
    0: [2, 3],
    1: [4],
}
```

#### 转化为显式 `edge_index`

`build_graph_with_derivation_edges()` 函数将 `derivation_edges` 转化为
PyTorch Geometric 格式的 `edge_index` 张量，每条派生边存储在元路径
`(src_type, "derives", dst_type)` 下：

```python
# 形状：(2, num_edges)
# 第 0 行：源节点（父节点）的类型内局部索引
# 第 1 行：目标节点（子节点）的类型内局部索引
data["research_background", "derives", "concept"].edge_index
# tensor([[0, 0],
#         [0, 1]])  # 根节点 0 → 子节点 0 和 1
```

这些 `edge_index` 张量被下游的**图-令牌接地模块（Graph-Token Grounding）**
直接消费，以将图结构信息注入多模态 LLM。

---

## 📂 核心数据结构与代码对应关系

| 概念 | Python 变量/函数 | 所在文件 |
|------|-----------------|---------|
| 系统提示词（步骤 0） | `x_sys_sk_0` | `chains.py` |
| 系统提示词（步骤 1+） | `x_sys_sk_1` | `chains.py` |
| 知识增强提示词 | `x_sys_ka` | `chains.py` |
| 步骤 0 提取链 | `create_iterative_skeleton_chain(llm, step=0)` | `chains.py` |
| 步骤 1+ 派生链 | `create_iterative_skeleton_chain(llm, step=1)` | `chains.py` |
| 知识增强链 | `create_knowledge_augmentation_chain(llm)` | `chains.py` |
| 迭代工作流函数 | `run_iterative_two_phase_graph_generation()` | `agent.py` |
| 派生关系字典 | `derivation_edges: dict[int, list[int]]` | `agent.py` |
| 迭代流水线 | `iterative_agent` | `agent.py` |
| 原始三阶段流水线 | `agent` | `agent.py` |
| 异构图构建（三阶段） | `build_hetero_graph_from_scaffold_keywords_v2()` | `pyg_utils.py` |
| 异构图构建（迭代式） | `build_graph_with_derivation_edges()` | `pyg_utils.py` |
| 全局节点索引 | `unified_idx`（节点字典中的字段） | `agent.py` / `pyg_utils.py` |
| 最终图对象 | `HeteroData`（PyTorch Geometric） | `pyg_utils.py` |

---

## 🚀 快速开始

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/sktsherlock/GraphAgent.git
cd GraphAgent

# 创建 conda 环境
conda create -n graphagent python=3.11
conda activate graphagent

# 安装推理所需依赖
pip install -r GraphAgent-inference/requirements.txt
```

### 获取预训练模型

我们在 🤗 Hugging Face 上提供以下预训练检查点：

- `GraphAgent/GraphAgent-8B`：多模态 Llama3，可接收图 Token 作为输入
- `GraphAgent/GraphTokenizer`：多模态图-文分词器
- `sentence-transformers/all-mpnet-base-v2`：文本图嵌入用的句子转换器

### 配置 API Key

GraphAgent 使用基于 API 的 LLM（默认为 `deepseek`）进行任务规划和图生成：

```bash
export OPENAI_API_KEY="your_api_key_here"
```

### 运行推理

```bash
bash GraphAgent-inference/run.sh

>>> Please enter a user instruction or file path (or type 'exit' to quit):
# 使用示例文件：
>>> GraphAgent-inference/demo/use_cases/teach_me_accelerate.txt
```

更多示例请查看 [use_cases](GraphAgent-inference/demo/use_cases) 目录。

---

## 📝 引用

如果本仓库对您的研究有帮助，请引用我们的论文：

```bibtex
@article{graphagent,
    title={GraphAgent: Agentic Graph Language Assistant},
    author={Yuhao Yang and Jiabin Tang and Lianghao Xia and Xingchen Zou and Yuxuan Liang and Chao Huang},
    year={2024},
    journal={arXiv preprint arXiv:2412.17029},
}
```
