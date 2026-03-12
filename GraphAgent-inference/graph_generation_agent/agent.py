from utils.storage import mem_store
from utils.llm import llm
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from .chains import (
    create_keywords_from_scaffold_text_chain,
    create_scaffold_text_parsing_chain,
    create_scaffold_node_extraction_chain,
    create_iterative_skeleton_chain,
    create_knowledge_augmentation_chain,
)
from functools import partial
import json
from .pyg_utils import build_hetero_graph_from_scaffold_keywords_v2, build_graph_with_derivation_edges
from colorama import Fore, Style

scaffold_node_extraction_chain = create_scaffold_node_extraction_chain(llm)
scaffold_text_parsing_chain = create_scaffold_text_parsing_chain(llm)
keywords_from_scaffold_text_chain = create_keywords_from_scaffold_text_chain(llm)

# Chains for the Iterative Two-Phase Graph Generation Workflow
# Step 0 uses x_sys_sk_0; step 1+ uses x_sys_sk_1; Phase 2 uses x_sys_ka.
iterative_skeleton_chain_step0 = create_iterative_skeleton_chain(llm, step=0)
iterative_skeleton_chain_step1 = create_iterative_skeleton_chain(llm, step=1)
knowledge_augmentation_chain = create_knowledge_augmentation_chain(llm)

def save_pyg_graph_to_mem(x):
    mem_store.mset([("pyg_graph", x["pyg_graph"])])
    return x

def parse_keywords_from_scaffold_texts(x, chain=keywords_from_scaffold_text_chain):
    scaffold_texts = x["scaffold_texts_parsing_output"]["scaffold_texts"]
    keywords = {}
    for item in scaffold_texts:
        keywords[item["node_id"]] = chain.invoke({"input": item["text"]})["keywords"]
    return keywords

def print_lambda(message, color=Fore.RESET):
    return lambda x: (print(color + message + json.dumps(x, indent=4) + Style.RESET_ALL), x)[1]


def run_iterative_two_phase_graph_generation(data_dict: dict, num_iterations: int = 1) -> dict:
    """
    Implements the Iterative Two-Phase Graph Generation Workflow described in the
    paper "GraphAgent: Agentic Graph Language Assistant".

    **Phase 1 – Skeleton Construction**

    * *Step 0*: ``iterative_skeleton_chain_step0`` (prompt ``x_sys_sk_0``) is
      called with the full ``knowledge_text`` and ``user_annotation`` to extract
      the initial set of top-level scaffold nodes.
    * *Steps 1 … num_iterations*: For every node produced in the previous step,
      ``iterative_skeleton_chain_step1`` (prompt ``x_sys_sk_1``) is called with
      that node's name and description text.  The resulting child nodes are
      linked back to their parent via the ``derivation_edges`` dictionary.

    **Phase 2 – Knowledge Augmentation**

    * ``knowledge_augmentation_chain`` (prompt ``x_sys_ka``) is invoked for
      every node collected in Phase 1 to enrich each node with a detailed
      description and a list of attributes.

    **Derivation tracking**

    Parent-child relationships are recorded in::

        derivation_edges: dict[int, list[int]]

    where the key is the parent node's ``unified_idx`` and the value is the list
    of its children's ``unified_idx`` values.  This mapping is later consumed by
    :func:`build_graph_with_derivation_edges` to create the explicit
    ``edge_index`` tensors used by the Graph-Token Grounding module.

    Args:
        data_dict: Must contain ``knowledge_text`` and ``user_annotation``.
        num_iterations: Number of iterative child-expansion steps (default 1).

    Returns:
        Updated ``data_dict`` augmented with:
        - ``all_skg_nodes``: flat list of every generated node (with
          ``unified_idx``, ``iteration``, and ``parent_unified_idx``).
        - ``derivation_edges``: dict mapping each parent's ``unified_idx`` to
          the list of its children's ``unified_idx`` values.
    """
    knowledge_text = data_dict.get("knowledge_text", "")
    user_annotation = data_dict.get("user_annotation", "")

    # ------------------------------------------------------------------
    # Phase 1, Step 0 – use x_sys_sk_0
    # ------------------------------------------------------------------
    print(Fore.CYAN + "ITERATIVE SKG – STEP 0 (x_sys_sk_0): extracting initial scaffold nodes…" + Style.RESET_ALL)
    step0_output = iterative_skeleton_chain_step0.invoke(
        {"knowledge_text": knowledge_text, "user_annotation": user_annotation}
    )
    initial_nodes: list[dict] = step0_output.get("scaffold_nodes", [])

    # Assign unified indices; root nodes have no parent.
    all_skg_nodes: list[dict] = []
    derivation_edges: dict[int, list[int]] = {}
    unified_idx = 0
    for node in initial_nodes:
        node["unified_idx"] = unified_idx
        node["iteration"] = 0
        node["parent_unified_idx"] = None
        all_skg_nodes.append(node)
        unified_idx += 1

    print(Fore.CYAN + "STEP 0 OUTPUT: " + json.dumps(initial_nodes, indent=4) + Style.RESET_ALL)

    current_iteration_nodes = list(initial_nodes)

    # ------------------------------------------------------------------
    # Phase 1, Steps 1+ – use x_sys_sk_1
    # ------------------------------------------------------------------
    for iteration in range(1, num_iterations + 1):
        print(
            Fore.CYAN
            + f"ITERATIVE SKG – STEP {iteration} (x_sys_sk_1): deriving child nodes…"
            + Style.RESET_ALL
        )
        next_iteration_nodes: list[dict] = []
        for parent_node in current_iteration_nodes:
            parent_description = parent_node.get("description", parent_node.get("name", ""))
            child_output = iterative_skeleton_chain_step1.invoke(
                {
                    "node_name": parent_node.get("name", ""),
                    "node_description": parent_description,
                }
            )
            child_nodes: list[dict] = child_output.get("scaffold_nodes", [])

            # Record parent → children derivation edges
            child_indices: list[int] = []
            for child_node in child_nodes:
                child_node["unified_idx"] = unified_idx
                child_node["iteration"] = iteration
                child_node["parent_unified_idx"] = parent_node["unified_idx"]
                all_skg_nodes.append(child_node)
                next_iteration_nodes.append(child_node)
                child_indices.append(unified_idx)
                unified_idx += 1

            if child_indices:
                derivation_edges[parent_node["unified_idx"]] = child_indices

        current_iteration_nodes = next_iteration_nodes

    # ------------------------------------------------------------------
    # Phase 2 – Knowledge Augmentation using x_sys_ka
    # ------------------------------------------------------------------
    print(Fore.GREEN + "ITERATIVE SKG – PHASE 2 (x_sys_ka): knowledge augmentation…" + Style.RESET_ALL)
    for node in all_skg_nodes:
        ka_output = knowledge_augmentation_chain.invoke(
            {
                "node_name": node.get("name", ""),
                "node_description": node.get("description", ""),
            }
        )
        node["augmented_description"] = ka_output.get("description", "")
        node["attributes"] = ka_output.get("attributes", [])

    print(
        Fore.GREEN
        + f"SKG CONSTRUCTION COMPLETE: {len(all_skg_nodes)} nodes, "
        + f"{sum(len(v) for v in derivation_edges.values())} derivation edges"
        + Style.RESET_ALL
    )
    return {
        **data_dict,
        "all_skg_nodes": all_skg_nodes,
        "derivation_edges": derivation_edges,
    }


def save_iterative_pyg_graph_to_mem(x):
    mem_store.mset([("pyg_graph", x["pyg_graph"])])
    return x

agent = (
    RunnablePassthrough.assign(scaffold_nodes_extraction_output=scaffold_node_extraction_chain)
    | print_lambda("SCAFFOLD NODES DISCOVERY: ", Fore.CYAN)
    | RunnablePassthrough.assign(scaffold_texts_parsing_output=scaffold_text_parsing_chain)
    | print_lambda("KNOWLEDGE AUGMENTATION: ", Fore.GREEN)
    | RunnablePassthrough.assign(keywords=partial(parse_keywords_from_scaffold_texts, chain=keywords_from_scaffold_text_chain))
    | print_lambda("SCAFFOLD NODES DISCOVERY-2 (KEYWORDS): ", Fore.MAGENTA)
    | print_lambda("GRAPH GROUNDING... ", Fore.YELLOW)
    | RunnablePassthrough.assign(pyg_graph=build_hetero_graph_from_scaffold_keywords_v2)
    | RunnableLambda(save_pyg_graph_to_mem)
    | RunnableLambda(lambda x: f"I have constructed a heterogeneous graph based on your request: {x.get('pyg_graph')}")
)

# ---------------------------------------------------------------------------
# Iterative Two-Phase agent pipeline
# ---------------------------------------------------------------------------
# This pipeline realises the workflow described in the paper:
#   1. run_iterative_two_phase_graph_generation  (Phase 1 + Phase 2)
#   2. build_graph_with_derivation_edges          (Graph-Token Grounding)
#   3. save to mem_store and return a status string
#
# num_iterations controls how many child-expansion steps are performed in
# Phase 1.  Use functools.partial to adjust the value at construction time:
#   e.g. partial(run_iterative_two_phase_graph_generation, num_iterations=2)
# ---------------------------------------------------------------------------
iterative_agent = (
    RunnableLambda(partial(run_iterative_two_phase_graph_generation, num_iterations=1))
    | print_lambda("ITERATIVE SKG NODES: ", Fore.CYAN)
    | RunnablePassthrough.assign(pyg_graph=build_graph_with_derivation_edges)
    | print_lambda("GRAPH GROUNDING (derivation edges)... ", Fore.YELLOW)
    | RunnableLambda(save_iterative_pyg_graph_to_mem)
    | RunnableLambda(
        lambda x: (
            f"I have constructed a Semantic Knowledge Graph via the Iterative "
            f"Two-Phase workflow: {x.get('pyg_graph')}"
        )
    )
)

# agent = (
#     # scaffold_node_extraction_chain
#     RunnablePassthrough.assign(scaffold_nodes_extraction_output=scaffold_node_extraction_chain)
#     | RunnableLambda(lambda x: (print("SCAFFOLD NODES DISCOVERY: ", json.dumps(x, indent=4)), x)[1])
#     | RunnablePassthrough.assign(scaffold_texts_parsing_output=scaffold_text_parsing_chain)
#     | RunnableLambda(lambda x: (print("KNOWLEDGE AUGMENTATION: ", json.dumps(x, indent=4)), x)[1])
#     | RunnablePassthrough.assign(keywords=partial(parse_keywords_from_scaffold_texts, chain=keywords_from_scaffold_text_chain))
#     | RunnableLambda(lambda x: (print("SCAFFOLD NODES DISCOVERY-2 (KEYWORDS): ", json.dumps(x, indent=4)), x)[1])
#     | RunnableLambda(lambda x: (print(Fore.YELLOW + "GRAPH GROUNDING... " + Style.RESET_ALL, x), x)[1])
#     | RunnablePassthrough.assign(pyg_graph=build_hetero_graph_from_scaffold_keywords_v2)
#     | RunnableLambda(save_pyg_graph_to_mem)
#     # | RunnableLambda(lambda x: {"output": x})
#     | RunnableLambda(lambda x: f"I have constructed a heterogeneous graph based on your request: {x.get('pyg_graph')}")
# )


if __name__ == "__main__":
    pass