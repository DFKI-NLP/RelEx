import torch
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
