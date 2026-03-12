from collections import defaultdict

import torch
from torch_geometric.data import Data, HeteroData




def build_hetero_graph_from_scaffold_keywords_v2(data_dict, add_edges_to_first_scaffold_node=False) -> HeteroData:
    
    scaffold_nodes: list[dict] = data_dict.get("scaffold_nodes_extraction_output")["scaffold_nodes"]
    scaffold_texts: list[dict] = data_dict.get("scaffold_texts_parsing_output")["scaffold_texts"]
    keywords_of_scaffold_nodes: dict[int, list[dict]] = data_dict.get("keywords")

    for i in range(len(scaffold_nodes)):
        scaffold_nodes[i]["description"] = scaffold_texts[i]["text"]

    num_scaffold_nodes = len(scaffold_nodes)
    num_keyword_nodes = sum([len(item) for item in keywords_of_scaffold_nodes.values()])

    idx_to_node = {}
    # create unified ID mapping for all nodes
    idx = 0
    for item in scaffold_nodes:
        item["unified_idx"] = idx
        idx_to_node[idx] = item
        idx += 1
    for item in keywords_of_scaffold_nodes:
        for keyword in keywords_of_scaffold_nodes[item]:
            # import pdb; pdb.set_trace()
            keyword["unified_idx"] = idx
            keyword["local_idx"] = idx - num_scaffold_nodes
            idx_to_node[idx] = keyword
            idx += 1
    assert idx == num_scaffold_nodes + num_keyword_nodes
            
    # group scaffold_nodes by node type to dictionary
    scaffold_nodes_by_type = defaultdict(list)
    # import pdb; pdb.set_trace()
    for node in scaffold_nodes:
        scaffold_nodes_by_type[node["type"]].append(node)
        # update scaffold node local index
        node.update({"local_idx":  len(scaffold_nodes_by_type[node["type"]]) - 1})

    data = HeteroData()
    # add scaffold nodes
    for node_type, nodes in scaffold_nodes_by_type.items():
        idx = [node["unified_idx"] for node in nodes]
        data[node_type].x = torch.zeros(len(idx), 1)
        data[node_type].unified_idx = torch.tensor(idx)
        data[node_type].description = [f"Type: {node['type']}; Name: {node['name']}; Description: {node['description']}" for node in nodes]
    
    keyword_node_idx = []
    keyword_node_description = []
    for keyword_nodes in keywords_of_scaffold_nodes.values():
        idx = [node["unified_idx"] for node in keyword_nodes]
        keyword_node_idx.extend(idx)
        keyword_node_description.extend([f"Name: {keyword_node['name']}, Description: {keyword_node['description']}" for keyword_node in keyword_nodes])
    data["keyword"].x = torch.zeros(len(keyword_node_idx), 1)
    data["keyword"].unified_idx = torch.tensor(keyword_node_idx)
    data["keyword"].description = keyword_node_description
    
    # create edges with meta paths
    
    # edges between anchor_scaffold_nodes and keywords
    for scaffold_node_id, keyword_nodes in keywords_of_scaffold_nodes.items():
        dst_nodes = [item["local_idx"] for item in keyword_nodes]
        src_nodes = [idx_to_node[scaffold_node_id]["local_idx"]] * len(dst_nodes)
        meta_path = [idx_to_node[scaffold_node_id]["type"], "has_keyword", "keyword"]
        edges = torch.stack([torch.tensor(src_nodes), torch.tensor(dst_nodes)], dim=0)
        if not hasattr(data[*meta_path], "edge_index"):
            data[*meta_path].edge_index = edges
        else:
            data[*meta_path].edge_index = torch.cat([data[*meta_path].edge_index, edges], dim=1)
    
    # edges between the first scaffold node and other scaffold nodes
    if add_edges_to_first_scaffold_node:
        first_scaffold_node = scaffold_nodes[0]
        src_node = [first_scaffold_node["local_idx"]]
        for i in range(1, len(scaffold_nodes)):
            meta_path = [first_scaffold_node["type"], "has_property", scaffold_nodes[i]["type"]]
            dst_node = [idx_to_node[i]["local_idx"]]
            edge = torch.stack([torch.tensor(src_node), torch.tensor(dst_node)], dim=0)
            if not hasattr(data[*meta_path], "edge_index"):
                data[*meta_path].edge_index = edge
            else:
                data[*meta_path].edge_index = torch.cat([data[*meta_path].edge_index, edge], dim=1)

    # append raw dict data to the HeteroData object
    data.scaffold_nodes_dlist = scaffold_nodes
    data.keyword_nodes_dict = keywords_of_scaffold_nodes
    return data


def build_graph_with_derivation_edges(data_dict) -> HeteroData:
    """
    Builds a HeteroData graph using the derivation-based edge construction
    described in the paper.

    The paper states: *"if a new node is generated from the textual description
    of a node in the previous iteration, we connect these two nodes."*

    The ``derivation_edges`` dict produced by
    :func:`run_iterative_two_phase_graph_generation` encodes exactly this
    parent-child relationship::

        derivation_edges: dict[int, list[int]]
        # key   = unified_idx of the parent node
        # value = list of unified_idx values of its child nodes

    This function converts that mapping into explicit ``edge_index`` tensors
    (shape ``(2, num_edges)``) stored in the returned ``HeteroData`` object
    under the meta-path ``(src_type, "derives", dst_type)``.  These tensors
    are consumed directly by the downstream Graph-Token Grounding module.

    Args:
        data_dict: Must contain:
            - ``all_skg_nodes``: flat list of all nodes produced by the
              Iterative Two-Phase workflow (each node must have ``unified_idx``,
              ``name``, and optionally ``type``, ``description``,
              ``augmented_description``).
            - ``derivation_edges``: dict mapping parent ``unified_idx`` →
              list of child ``unified_idx`` values.

    Returns:
        A ``HeteroData`` graph with node features, descriptions, and
        derivation-based ``edge_index`` tensors per meta-path.
    """
    all_skg_nodes: list[dict] = data_dict.get("all_skg_nodes", [])
    derivation_edges: dict[int, list[int]] = data_dict.get("derivation_edges", {})

    # Build unified-index → node mapping for edge construction
    idx_to_node: dict[int, dict] = {node["unified_idx"]: node for node in all_skg_nodes}

    # Group nodes by type and assign local (type-scoped) indices
    nodes_by_type: defaultdict = defaultdict(list)
    for node in all_skg_nodes:
        node_type = node.get("type", "concept")
        node["local_idx"] = len(nodes_by_type[node_type])
        nodes_by_type[node_type].append(node)

    data = HeteroData()

    # Add node features and descriptions
    for node_type, nodes in nodes_by_type.items():
        data[node_type].x = torch.zeros(len(nodes), 1)
        data[node_type].unified_idx = torch.tensor([node["unified_idx"] for node in nodes])
        data[node_type].description = [
            f"Type: {node.get('type', 'concept')}; Name: {node['name']}; "
            f"Description: {node.get('augmented_description', node.get('description', ''))}"
            for node in nodes
        ]

    # Convert derivation_edges to edge_index tensors grouped by meta-path.
    # Meta-path format: (src_type, "derives", dst_type)
    for parent_unified_idx, child_unified_indices in derivation_edges.items():
        parent_node = idx_to_node.get(parent_unified_idx)
        if parent_node is None:
            continue
        src_type = parent_node.get("type", "concept")

        for child_unified_idx in child_unified_indices:
            child_node = idx_to_node.get(child_unified_idx)
            if child_node is None:
                continue
            dst_type = child_node.get("type", "concept")
            edge = torch.tensor(
                [[parent_node["local_idx"]], [child_node["local_idx"]]]
            )
            if not hasattr(data[src_type, "derives", dst_type], "edge_index"):
                data[src_type, "derives", dst_type].edge_index = edge
            else:
                data[src_type, "derives", dst_type].edge_index = torch.cat(
                    [data[src_type, "derives", dst_type].edge_index, edge], dim=1
                )

    # Store raw data for downstream use
    data.all_skg_nodes = all_skg_nodes
    data.derivation_edges = derivation_edges
    return data