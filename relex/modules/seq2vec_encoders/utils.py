from typing import List, Tuple

import torch
import numpy as np
from collections import deque
from allennlp.nn.util import masked_max, masked_mean, get_final_encoder_states


def pool(
    vector: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
    pooling: str,
    is_bidirectional: bool,
) -> torch.Tensor:
    if pooling == "max":
        return masked_max(vector, mask, dim)
    elif pooling == "mean":
        return masked_mean(vector, mask, dim)
    elif pooling == "sum":
        return torch.sum(vector, dim)
    elif pooling == "final":
        return get_final_encoder_states(vector, mask, is_bidirectional)
    else:
        raise ValueError(f"'{pooling}' is not a valid pooling operation.")


class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    Code taken from
    https://github.com/qipeng/gcn-over-pruned-trees/blob/master/model/tree.py
    """

    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = []
        self.head = None
        self.dep_label = None
        self.token = None

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if hasattr(self, "_size"):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def is_root(self):
        return self.head == 0

    def parent(self):
        return self.parent

    def parent_idx(self):
        return self.head

    def dep_label(self):
        return self.dep_label

    def depth(self):
        if hasattr(self, "_depth"):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

    def __repr__(self):
        if hasattr(self, "_repr"):
            return self._repr
        else:
            self._repr = f'{self.token}-{self.idx}[{self.dep_label}->{self.head}]'

def dep_heads_to_tree(
    dep_heads: List[int],
    length: int,
    head: Tuple[int, int],
    tail: Tuple[int, int],
    prune: int = -1,
    dep_labels: List[str] = None,
    tokens: List[str] = None,
) -> Tree:
    """
    Convert a sequence of dependency head indexes into a Tree.
    """
    root = None

    if prune < 0:
        nodes = [Tree() for _ in dep_heads]

        for i in range(len(nodes)):
            h = dep_heads[i]
            nodes[i].idx = i
            nodes[i].head = h
            if dep_labels is not None:
                nodes[i].dep_label = dep_labels[i]
            if tokens is not None:
                nodes[i].token = tokens[i]
            nodes[i].dist = -1  # just a filler
            if h == 0:
                root = nodes[i]
            else:
                nodes[h - 1].add_child(nodes[i])
    else:
        # find dependency path
        head_start, head_end = head
        tail_start, tail_end = tail
        subj_pos = list(range(head_start, head_end + 1))
        obj_pos = list(range(tail_start, tail_end + 1))

        cas = None

        subj_ancestors = set(subj_pos)
        for s in subj_pos:
            h = dep_heads[s]
            tmp = [s]
            while h > 0:
                tmp += [h - 1]
                subj_ancestors.add(h - 1)
                h = dep_heads[h - 1]

            if cas is None:
                cas = set(tmp)
            else:
                cas.intersection_update(tmp)

        obj_ancestors = set(obj_pos)
        for o in obj_pos:
            h = dep_heads[o]
            tmp = [o]
            while h > 0:
                tmp += [h - 1]
                obj_ancestors.add(h - 1)
                h = dep_heads[h - 1]
            cas.intersection_update(tmp)

        # find lowest common ancestor
        if len(cas) == 1:
            lca = list(cas)[0]
        else:
            child_count = {k: 0 for k in cas}
            for ca in cas:
                if dep_heads[ca] > 0 and dep_heads[ca] - 1 in cas:
                    child_count[dep_heads[ca] - 1] += 1

            # the LCA has no child in the CA set
            for ca in cas:
                if child_count[ca] == 0:
                    lca = ca
                    break

        path_nodes = subj_ancestors.union(obj_ancestors).difference(cas)
        path_nodes.add(lca)

        # compute distance to path_nodes
        dist = [-1 if i not in path_nodes else 0 for i in range(length)]

        for i in range(length):
            if dist[i] < 0:
                stack = [i]
                while stack[-1] >= 0 and stack[-1] not in path_nodes:
                    stack.append(dep_heads[stack[-1]] - 1)

                if stack[-1] in path_nodes:
                    for d, j in enumerate(reversed(stack)):
                        dist[j] = d
                else:
                    for j in stack:
                        if j >= 0 and dist[j] < 0:
                            dist[j] = int(1e4)  # aka infinity

        highest_node = lca
        nodes = [Tree() if dist[i] <= prune else None for i in range(length)]

        for i in range(len(nodes)):
            if nodes[i] is None:
                continue
            h = dep_heads[i]
            nodes[i].idx = i
            nodes[i].dist = dist[i]
            nodes[i].head = h
            if dep_labels is not None:
                nodes[i].dep_label = dep_labels[i]
            if tokens is not None:
                nodes[i].token = tokens[i]
            if h > 0 and i != highest_node:
                assert nodes[h - 1] is not None
                nodes[h - 1].add_child(nodes[i])

        root = nodes[highest_node]

    assert root is not None
    return root


def tree_to_adjacency_list(
    tree: Tree, directed: bool = True, add_self_loop: bool = False
) -> List[Tuple[int, int]]:
    """
    Convert a tree object to an adjacency list.
    """
    adjacency_list = []

    queue = deque([tree])
    idx = []
    while queue:
        t = queue.popleft()

        idx += [t.idx]

        for c in t.children:
            adjacency_list.append((t.idx, c.idx))
        queue += t.children

    if not directed:
        inverse_edges = []
        for edge in adjacency_list:
            head, tail = edge
            inverse_edges.append((tail, head))
        adjacency_list.extend(inverse_edges)

    if add_self_loop:
        for i in idx:
            adjacency_list.append((i, i))

    return adjacency_list
