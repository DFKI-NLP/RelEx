from enum import Enum
from typing import List

import torch

from allennlp.nn import util
from allennlp.nn.util import masked_max, masked_mean, get_final_encoder_states


class PoolingScope(Enum):
    SEQUENCE = "sequence"
    HEAD = "head"
    TAIL = "tail"


def pool(vector: torch.Tensor,
         mask: torch.Tensor,
         dim: int,
         pooling: str,
         is_bidirectional: bool) -> torch.Tensor:
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


def scoped_pool(tokens: torch.Tensor,
                mask: torch.Tensor,
                pooling: str,
                pooling_scopes: List[PoolingScope],
                is_bidirectional: bool = False,
                head: torch.Tensor = None,
                tail: torch.Tensor = None) -> torch.Tensor:
    pooling_masks = []

    if PoolingScope.SEQUENCE in pooling_scopes:
        pooling_masks.append(mask.unsqueeze(-1))

    if PoolingScope.HEAD in pooling_scopes or PoolingScope.TAIL in pooling_scopes:
        assert head is not None and tail is not None, \
            "head and tail offsets are required for pooling on entities"

        batch_size, seq_len, _ = tokens.size()
        pos_range = util.get_range_vector(
                seq_len, util.get_device_of(tokens)).repeat((batch_size, 1))

        if PoolingScope.HEAD in pooling_scopes:
            head_start = head[:, 0].unsqueeze(dim=1)
            head_end = head[:, 1].unsqueeze(dim=1)
            head_mask = ((torch.ge(pos_range, head_start)
                          * torch.le(pos_range, head_end)).unsqueeze(-1).long())
            pooling_masks.append(head_mask)

        if PoolingScope.TAIL in pooling_scopes:
            tail_start = tail[:, 0].unsqueeze(dim=1)
            tail_end = tail[:, 1].unsqueeze(dim=1)

            tail_mask = ((torch.ge(pos_range, tail_start)
                          * torch.le(pos_range, tail_end)).unsqueeze(-1).long())
            pooling_masks.append(tail_mask)

    assert pooling_masks, "At least one pooling scope must be defined"

    pooled = [pool(tokens, mask, dim=1, pooling=pooling,
                   is_bidirectional=is_bidirectional) for mask in pooling_masks]

    return torch.cat(pooled, dim=-1)
