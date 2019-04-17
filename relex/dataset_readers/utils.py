from typing import List, Tuple

import networkx as nx


def parse_adjacency_indices(dep: List[str],
                            dep_heads: List[int],
                            head: Tuple[int, int],
                            tail: Tuple[int, int],
                            pruning_distance: int = 1):
    dep_tree = nx.DiGraph()
    dep_tree.add_nodes_from(range(len(dep)))
    for (node_idx, dep_head) in enumerate(dep_heads):
        dep_tree.add_edge(node_idx, node_idx)
        if dep_head > 0:
            dep_tree.add_edge(dep_head - 1, node_idx)
            dep_tree.add_edge(node_idx, dep_head - 1)

    dep_edges = dep_tree.edges
    if pruning_distance > -1:
        pruned_nodes = []
        sdp_nodes = nx.shortest_path(dep_tree, head[0], tail[0])

        for node in dep_tree.nodes:
            for sdp_node in sdp_nodes:
                if node in pruned_nodes:
                    continue

                if (node == sdp_node or
                        node in range(head[0], head[1] + 1) or
                        node in range(tail[0], tail[1] + 1)):
                    pruned_nodes.append(node)
                else:
                    path_length = nx.shortest_path_length(dep_tree, node, sdp_node)
                    if path_length <= pruning_distance:
                        pruned_nodes.append(node)

        dep_edges = [edge for edge in dep_edges
                     if edge[0] in pruned_nodes and edge[1] in pruned_nodes]

    return dep_edges
