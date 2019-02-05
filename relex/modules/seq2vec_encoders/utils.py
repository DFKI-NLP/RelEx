import torch
import numpy as np
from allennlp.nn.util import masked_max, masked_mean


def position_encoding_init(n_position: int, embedding_dim: int):
    position_enc = np.array(
        [
            [
                pos / np.power(10000, 2 * (j // 2) / embedding_dim)
                for j in range(embedding_dim)
            ]
            if pos != 0
            else np.zeros(embedding_dim)
            for pos in range(n_position)
        ]
    )
    position_enc[1:, 0::2] = np.sin(
        position_enc[1:, 0::2]
    )  # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(
        position_enc[1:, 1::2]
    )  # apply cos on 1st,3rd,5th...emb_dim
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def pool(
    vector: torch.Tensor, mask: torch.Tensor, dim: int, pooling: str
) -> torch.Tensor:
    if pooling == "max":
        return masked_max(vector, mask, dim)
    elif pooling == "mean":
        return masked_mean(vector, mask, dim)
    elif pooling == "sum":
        return torch.sum(vector, dim)
    else:
        raise ValueError(f"'{pooling}' is not a valid pooling operation.")
