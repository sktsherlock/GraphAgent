<div align="center">
    <img src="assets/cover_pic.jpg" height="300px">
</div>

<h1 align="center">GraphAgent: Agentic Graph Language Assistant</h1>

<p align="center">
📖 <a href="https://arxiv.org/abs/2412.17029" target="_blank"> Paper at Arxiv</a> · 🤗 <a href="https://huggingface.co/GraphAgent/GraphAgent-7B" target="_blank">GraphAgent Model</a> · 🤗 <a href="https://huggingface.co/GraphAgent/GraphTokenizer" target="_blank">Graph Tokenizer Model</a> 
 · 🤗 <a href="https://huggingface.co/datasets/GraphAgent/GraphAgent-Datasets" target="_blank">GraphAgent Datasets</a>
</p>

<p align="center">
🇨🇳 <a href="README_zh.md">中文文档 / Chinese README</a>
</p>


## 📋 To-Do List
- [x] Release inference code
- [x] Release model checkpoints
- [x] Release training and evaluation datasets
- [x] Release training code

## 🌟 Overview

Real-world data is represented in both structured (e.g., graph connections) and unstructured (e.g., textual, visual information) formats, encompassing complex relationships that include explicit links (such as social connections and user behaviors) and implicit interdependencies among semantic entities, often illustrated through knowledge graphs. In this work, we propose GraphAgent, an automated agent pipeline that addresses both explicit graph dependencies and implicit graph-enhanced semantic inter-dependencies, aligning with practical data scenarios for predictive tasks (e.g., node classification) and generative tasks (e.g., text generation). GraphAgent comprises three key components: (i) a Graph Generator Agent that builds knowledge graphs to reflect complex semantic dependencies; (ii) a Task Planning Agent that interprets diverse user queries and formulates corresponding tasks through agentic self-planning; and (iii) a Task Execution Agent that efficiently executes planned tasks while automating tool matching and invocation in response to user queries. These agents collaborate seamlessly, integrating language models with graph language models to uncover intricate relational information and data semantic dependencies. Through extensive experiments on various graph-related predictive and text generative tasks on diverse datasets, we demonstrate the effectiveness of our GraphAgent across various settings.

## 🚀 Getting Started

### Invoking GraphAgent (Inference)
#### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GraphAgent.git
cd GraphAgent

# Create a conda environment
conda create -n graphagent python=3.11
conda activate graphagent

# Install requirements for GraphAgent inference
pip install -r GraphAgent-inference/requirements.txt
```

#### Get Pre-trained Models

We provide several pre-trained checkpoints on 🤗 Hugging Face to power the full potential of GraphAgent:

- `GraphAgent/GraphAgent-8B`: Graph action model for GraphAgent as a multimodal llama3 that can take graph tokens as input.
- `GraphAgent/GraphTokenizer`: A multimodal graph-text tokenizer for tokenizing graphs into continuous tokens.
- `sentence-transformers/all-mpnet-base-v2`: The sentence transformer for text graph embedding.

You can download these checkpoints to a local dir and replace them in `GraphAgent-inference/run.sh`. Or, the program would also automatically download them for you.

#### Set the Planner and API Token

We utilize API-based LLM calls for task planning and graph generation. The default planner here is `deepseek`, where you can find in `GraphAgent-inference/run.sh`. Put your API key in 
```bash
export OPENAI_API_KEY=""
```
that is corresponding to the planner.

#### Inference Examples

```bash
bash GraphAgent-inference/run.sh

>>> Please enter a user instruction or file path (or type 'exit' to quit):

# use GraphAgent-inference/demo/use_cases/teach_me_accelerate.txt as an example
>>> Please enter a user instruction or file path (or type 'exit' to quit): GraphAgent-inference/demo/use_cases/teach_me_accelerate.txt
```

Then you will have a close look on how GraphAgent works to achieve your task.

For more detailed and diverse examples on what GraphAgent can do for you, check out our [use_cases](GraphAgent-inference/demo/use_cases) directory.

<!-- ## 📖 Documentation

### Model Inference


```python
# Load specific model variant
model = GraphAgent.from_pretrained("graphagent-base")
```

#### Input Formats

GraphAgent supports multiple input formats:
- NetworkX graphs
- PyTorch Geometric Data objects
- DGL graphs
- Edge list formats

```python
# Using NetworkX
import networkx as nx
G = nx.random_geometric_graph(200, 0.125)
result = model.predict(G)

# Using edge lists
edges = [(0, 1), (1, 2), (2, 3)]
result = model.predict_from_edges(edges)
``` -->

### GraphAgent Dataset (Coming Soon!)
| | IMDB | ACM | Arxiv-Papers | ICLR-Peer Reviews | Related Work Generation | GovReport Summarization |
|---|---|---|---|---|---|---|
| Task Type | Predictive | Predictive | Predictive | Predictive | Generative | Generative |
| Sub-Task | NC | NC | Paper Classification | Paper Judgement Prediction | Text Generation | Text Summarization |
| Pre-defined Graph? | ✓ | ✓ | × | × | × | × |
| #Train Samples | 2,400 | - | 5,175 | 3,141 | 4,155 | - |
| #Eval Samples | - | 1000 | 500 | 500 | 500 | 304 |
| #Tokens | 10M | 0.8M | 30M | 45M | 93M | 2M |
| #Pre-defined Graph Nodes | 11,616 | 10,942 | - | - | - | - |
| SKG Source | People Entities | Paper | Paper | Paper, Reviews | Multiple Papers | Documents |
| #SKG Nodes | 57,120 | 20,388 | 153,555 | 161,592 | 875,921 | 15,621 |


### Training GraphAgent with Your Own Data (Coming Soon!)

The training code and procedures will be released in future updates. Stay tuned!

## 📊 Benchmarks

#### Zero-shot classification task on ACM-1000
| Metric | Trained on | SAGE | GAT | HAN | HGT | HetGNN | HiGPT | GraphAgent | Imprv. |
|---|---|---|---|---|---|---|---|---|---|
| Micro-F1 (%) | IMDB-1 | 32.93±4.18 | 35.67±0.53 | 34.07±1.11 | 32.40±0.14 | 37.43±4.34 | 45.40±0.89 | **51.21±1.32** | 12.8% |
| | IMDB-40 | 31.73±0.05 | 23.93±1.44 | 26.97±1.94 | 35.60±0.99 | 31.80±0.16 | 50.50±0.77 | **74.98±1.24** | 48.5% |
| Macro-F1 (%) | IMDB-1 | 26.47±2.69 | 29.08±1.31 | 22.50±4.16 | 16.31±0.05 | 31.39±4.68 | 41.77±1.24 | **46.82±1.43** | 12.1% |
| | IMDB-40 | 31.17±0.17 | 21.41±0.71 | 23.13±1.32 | 27.49±1.22 | 31.44±0.17 | 45.85±0.89 | **74.98±1.12** | 63.5% |
| AUC (%) | IMDB-1 | 49.34±2.47 | 52.48±0.38 | 51.28±0.86 | 50.00±0.00 | 53.18±2.95 | 59.69±0.82 | **64.10±1.25** | 7.4% |
| | IMDB-40 | 48.67±0.13 | 43.20±1.08 | 45.45±1.46 | 51.48±0.43 | 48.72±0.06 | 63.60±0.51 | **80.90±1.01** | 27.2% |

#### Complex graph predictive tasks on Arxiv-Papers and ICLR-Peer Reviews
| Method | Model Size | Arxiv-Papers |  |  | ICLR-Peer Reviews |  |  |
|---|---|---|---|---|---|---|---|
| | | Mi-F1 | Ma-F1 | AUC | Mi-F1 | Ma-F1 | AUC |
| **Open-sourced LLMs** | | | | | | | |
| Llama3-8b | 8B | 0.514 | 0.289 | 0.527 | 0.402 | 0.394 | 0.502 |
| Mistral-Nemo | 12B | 0.510 | 0.292 | 0.615 | 0.272 | 0.246 | 0.380 |
| Llama3-70b | 70B | 0.630 | 0.330 | 0.635 | 0.434 | 0.421 | 0.551 |
| Qwen2-72b | 72B | 0.632 | 0.472 | 0.700 | 0.344 | 0.277 | 0.509 |
| **API-based Commercial LLMs** | | | | | | | |
| Deepseek-Chat-V2 | 236B→21B | 0.746 | 0.580 | 0.757 | 0.362 | 0.312 | 0.516 |
| GPT4o-mini | - | 0.592 | 0.343 | 0.634 | **0.692*** | 0.592 | 0.591 |
| Gemini-1.5-Flash | - | 0.748 | 0.504 | 0.714 | 0.684 | 0.487 | 0.533 |
| **Finetuned LLMs** | | | | | | | |
| Llama3-8b Finetuned | 8B | 0.794 | 0.593 | 0.736 | 0.620 | 0.554 | 0.553 |
| **GraphRAG Implementations** | | | | | | | |
| Llama3-8b + GraphRAG | 8B | 0.516 | 0.288 | 0.601 | 0.430 | 0.427 | 0.517 |
| Llama3-70b + GraphRAG | 70B | 0.603 | 0.324 | 0.623 | 0.308 | 0.296 | 0.401 |
| GraphAgent-Task Expert | 8B | 0.820 | 0.620 | 0.768 | 0.686 | **0.620*** | **0.615*** |
| GraphAgent-General | 8B | **0.840*** | **0.621*** | **0.769*** | 0.667 | 0.604 | 0.607 |
| GraphAgent-Zero-Shot | 8B | 0.739 | 0.512 | 0.701 | 0.538 | 0.531 | 0.563 |

#### Content generation on ACL-EMNLP related work instructions.
| Method | Model Size | PPL-Llama3-70b |  | PPL-Qwen2-72b |  |
|---|---|---|---|---|---|
| | | Mean | Max | Mean | Max |
| **Open-sourced LLMs** | | | | | |
| Llama3-8b | 8B | 7.016 | 13.061 | 7.491 | 12.787 |
| Mistral-Nemo | 12B | 7.367 | 15.967 | 6.872 | 12.065 |
| Llama3-70b | 70B | 6.168 | 14.436 | 5.877 | 12.897 |
| Qwen2-72b | 72B | 6.043 | 11.675 | 5.325 | 11.302 |
| **API-based Commercial LLMs** | | | | | |
| Deepseek-Chat-V2 | 236B→21B | 5.632 | 13.483 | 5.144 | 10.337 |
| GPT4o-mini | - | 7.277 | 15.480 | 6.818 | 13.267 |
| Gemini-1.5-Flash | - | 5.188 | 10.399 | 5.377 | 10.779 |
| **Finetuned LLMs** | | | | | |
| Llama3-8b Finetuned | 8B | 7.682 | 19.452 | 7.629 | 18.757 |
| **GraphRAG Implementations** | | | | | |
| Llama3-8b + GraphRAG | 8B | 7.098 | 18.092 | 6.539 | 14.722 |
| Llama3-70b + GraphRAG | 70B | 6.590 | 14.827 | 6.135 | 14.163 |
| GraphAgent-Task Expert | 8B | 3.805 | 10.316 | 4.069 | 11.685 |
| GraphAgent-General | 8B | **3.618*** | **8.000*** | **3.867*** | **8.775*** |


## 🇨🇳 中文说明：GraphAgent 如何自动构建语义知识图谱

GraphAgent 通过一套全自动的多智能体流水线来构建**语义知识图谱（Semantic Knowledge Graph, SKG）**。整个过程分为三个核心组件协同完成：

### 一、三大核心组件

| 组件 | 职责 |
|---|---|
| **图生成智能体（Graph Generator Agent）** | 从非结构化文本中自动构建语义知识图谱，反映复杂的语义依赖关系 |
| **任务规划智能体（Task Planning Agent）** | 解析用户查询，通过自主规划将查询分解为可执行的子任务 |
| **任务执行智能体（Task Execution Agent）** | 调用相应工具执行规划好的任务，完成预测或生成任务 |

### 二、语义知识图谱的自动构建流程（迭代两阶段工作流）

图生成智能体采用**迭代两阶段工作流**来完成 SKG 的自动构建：

#### 第一阶段：骨架构建（Skeleton Construction）

**步骤 0 — 初始脚手架节点提取**

使用系统提示 `x_sys_sk_0`，以用户输入的 `knowledge_text`（知识文本）和 `user_annotation`（用户标注/任务描述）为输入，由大语言模型（LLM）自动提取出一组**顶层抽象概念节点**（即脚手架节点，scaffold nodes）。这些节点代表文本中最高层级的关键主题或实体。

**步骤 1 … N — 迭代子节点推导**

使用系统提示 `x_sys_sk_1`，对上一步生成的每个父节点，LLM 根据该父节点的描述文本生成更具体、更细粒度的**子节点**，并将父子关系记录为 `derivation_edges`（推导边）字典。此过程可迭代多轮，逐层细化图结构。

#### 第二阶段：知识增强（Knowledge Augmentation）

使用系统提示 `x_sys_ka`，对第一阶段生成的所有节点，LLM 依次为每个节点补充**详细描述**和**关键属性列表**，从而使每个节点携带丰富的语义信息。

#### 图的落地与标记化（Graph Grounding & Tokenization）

1. **图构建**：调用 `build_graph_with_derivation_edges` 函数，将所有节点和推导边转化为 `HeteroData` 异构图（`torch_geometric` 格式）。节点按类型分组并分配局部索引，推导边形成元路径 `(src_type, "derives", dst_type)` 的 `edge_index` 张量。

2. **图标记化**：`hetero_graph_tokenize` 将异构图编码为连续的图 token，供后续的图动作智能体（Graph Action Agent）输入多模态 LLM，最终完成节点分类、文本生成等下游任务。

### 三、完整流水线示意

```
用户输入 (文本 + 任务说明)
       │
       ▼
任务规划智能体  →  解析查询，生成 knowledge_text / user_annotation
       │
       ▼
图生成智能体
  ├─ [Phase 1, Step 0]  x_sys_sk_0  →  顶层脚手架节点
  ├─ [Phase 1, Step 1+] x_sys_sk_1  →  迭代推导子节点 + derivation_edges
  └─ [Phase 2]          x_sys_ka    →  知识增强（描述 + 属性）
       │
       ▼
图落地  build_graph_with_derivation_edges  →  HeteroData 异构图
       │
       ▼
图标记化  hetero_graph_tokenize  →  图 token 序列
       │
       ▼
图动作智能体（多模态 LLM）  →  最终预测 / 生成结果
```

### 四、关键设计亮点

- **全自动**：无需人工标注图结构，LLM 从原始文本中端到端地抽取节点、构建边、增强知识。
- **迭代细化**：通过多轮子节点推导，图结构从粗粒度到细粒度逐步完善，层次清晰。
- **异构图**：节点和边均带有类型信息，支持复杂的元路径查询和图神经网络处理。
- **任务自适应**：脚手架节点的抽取策略会根据任务类型（预测任务 vs. 生成任务）自动调整，确保图的语义与下游任务紧密对齐。

---

## 🇨🇳 中文说明：节点文本描述的存储、关联与 LLM 使用方式

### 一、节点文本描述存储在哪里？

LLM 生成的每个知识图谱节点的文本描述，以 **Python 列表**的形式直接存储在 `torch_geometric` 的 `HeteroData` 对象（即 `pyg_graph`）中，字段名为 `.description`。

具体而言，图构建函数（`build_hetero_graph_from_scaffold_keywords_v2` 或 `build_graph_with_derivation_edges`）在创建 `HeteroData` 时，会为每种节点类型写入如下格式的描述字符串列表：

```python
# 脚手架节点（scaffold node）示例
data[node_type].description = [
    "Type: paper_contribution; Name: Mamba_model; Description: This paper introduces Mamba...",
    ...
]

# 关键词节点（keyword node）示例
data["keyword"].description = [
    "Name: selective_state_space; Description: A mechanism that...",
    ...
]
```

此时节点的特征向量 `.x` 仅为全零占位张量（`torch.zeros(num_nodes, 1)`），真正的语义特征尚未填入。

### 二、文本描述如何与知识图谱建立联系？

文本描述与图结构的绑定分两步完成：

**步骤 1 — 文本编码（`encode_node_text`）**

在图标记化阶段（`graph_tokenizer.py`），`encode_node_text` 函数遍历每种节点类型，使用预训练的 `SentenceTransformer`（默认为 `all-mpnet-base-v2`）将每个描述字符串编码为 768 维的稠密语义向量，并**覆盖**原来的零占位张量，写回 `pyg_graph[node_type].x`：

```python
for node_type in pyg_graph.node_types:
    for i, node_text in enumerate(node_set["description"]):
        node_text_emb = sentence_transformer.encode([node_text], ...)
        x_dict_type_i[i] = node_text_emb   # 768 维文本嵌入
    node_set["x"] = x_dict_type_i           # 写回节点特征矩阵
```

**步骤 2 — 图神经网络聚合（`MetaHGTConv`）**

768 维文本嵌入随即通过 **MetaHGT**（元异构图 Transformer）进行消息传递。MetaHGT 以节点类型名称和边类型名称的文本嵌入作为元参数，动态生成每种类型的注意力权重矩阵，将**图的拓扑结构信息**融合进每个节点的向量表示：

```python
res = metahgt_model(
    x_dict=pyg_graph.x_dict,           # 各类型节点文本嵌入
    edge_index_dict=pyg_graph.edge_index_dict,  # 图结构（推导边等）
    node_type_feas_dict=...,            # 节点类型名称的嵌入
    edge_type_feas_dict=...,            # 边类型名称的嵌入
)
pyg_graph.x_dict = res   # 聚合后的节点嵌入（同样为出通道维度）
```

经过这一步，每个节点的嵌入不仅包含自身的文本语义，还融合了邻居节点的语义和图的结构信息。

### 三、节点嵌入如何让 LLM 能够使用？

图嵌入通过**图 Token 插入**机制注入 LLM 的输入序列，整个过程如下：

**① 构建带占位符的文本提示（`build_prompt`）**

`build_input.py` 为每种节点类型在提示文本中插入一个 `<graph>` 占位符：

```
System: You are a powerful AI assistant...
User: <user_instruction>
Heterogeneous Knowledge Graph:
  "concept" nodes: <graph>; "keyword" nodes: <graph>; ...
```

**② 将 `<graph>` 展开为 patch token 序列（`preprocess_graph_Hetero`）**

该函数将每个 `<graph>` 替换为若干 `<g_patch>` token（数量等于该类型节点的数量），并在首尾分别加上 `<g_start>` / `<g_end>` 定界符：

```
<g_start><g_patch><g_patch>...<g_patch><g_end>
          ←  num_nodes 个 patch token  →
```

**③ 在 LLM 的 forward 中注入图嵌入（`graphllm.py`）**

`HeteroGraphLLMModel.forward` 在处理 embedding 序列时，找到每个 `<g_start>` 位置，将对应的节点嵌入通过线性投影层（`graph_projector: nn.Linear(graph_hidden_size, llm_hidden_size)`）映射到 LLM 隐层维度，再**原位替换** `<g_patch>` token 的 embedding：

```python
# 每个 <g_patch> 位置的 embedding 被替换为对应的节点嵌入（经 projector 变换）
cur_new_input_embeds = torch.cat([
    cur_input_embeds[: graph_start_token_pos + 1],
    cur_graph_features,          # ← 投影后的节点嵌入向量组
    cur_input_embeds[graph_start_token_pos + num_patches + 1 :],
], dim=0)
```

这样，LLM 在自回归生成时，就能像处理文本 token 一样，通过注意力机制同时"看到"图节点的语义信息和原始用户指令，从而完成下游的预测或生成任务。

### 四、完整数据流总结

```
LLM 生成节点描述（字符串）
          │
          ▼  存储于 pyg_graph[node_type].description（列表）
          │
          ▼  SentenceTransformer 编码 → 768 维向量
          │  写入 pyg_graph[node_type].x
          │
          ▼  MetaHGTConv 图神经网络聚合（融合邻居 + 图结构）
          │  更新 pyg_graph.x_dict
          │
          ▼  graph_projector 线性映射（768/GNN输出 → LLM隐层维度）
          │
          ▼  原位替换 <g_patch> token 的 embedding
          │
          ▼  LLM 注意力机制统一处理文本 token + 图节点嵌入
          │
          ▼  生成最终回答（分类标签 / 生成文本）
```

---

## 📝 Citation

If you find this repository useful, please cite our paper:

```bibtex

@article{graphagent,
      title={GraphAgent: Agentic Graph Language Assistant}, 
      author={Yuhao Yang and Jiabin Tang and Lianghao Xia and Xingchen Zou and Yuxuan Liang and Chao Huang},
      year={2024},
      journal={arXiv preprint arXiv:2412.17029},
}
```
