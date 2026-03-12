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
