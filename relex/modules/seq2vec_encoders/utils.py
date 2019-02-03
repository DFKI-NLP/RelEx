import torch
import numpy as np


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


def sum_pool(x, lengths):
    out = torch.FloatTensor(x.size(1), x.size(2)).zero_()  # BxF
    for i in range(x.size(1)):
        out[i] = torch.sum(x[: lengths[i], i, :], 0)
    return out


def mean_pool(x, lengths):
    out = torch.FloatTensor(x.size(1), x.size(2)).zero_()  # BxF
    for i in range(x.size(1)):
        out[i] = torch.mean(x[: lengths[i], i, :], 0)
    return out


def max_pool(x, lengths):
    out = torch.FloatTensor(x.size(1), x.size(2)).zero_()  # BxF
    for i in range(x.size(1)):
        out[i, :] = torch.max(x[: lengths[i], i, :], 0)[0]
    return out


def min_pool(x, lengths):
    out = torch.FloatTensor(x.size(1), x.size(2)).zero_()  # BxF
    for i in range(x.size(1)):
        out[i] = torch.min(x[: lengths[i], i, :], 0)[0]
    return out


def hier_pool(x, lengths, n=5):
    out = torch.FloatTensor(x.size(1), x.size(2)).zero_()  # BxF
    if x.size(0) <= n:
        return mean_pool(x, lengths)  # BxF
    for i in range(x.size(1)):
        sliders = []
        if lengths[i] <= n:
            out[i] = torch.mean(x[: lengths[i], i, :], 0)
            continue
        for j in range(lengths[i] - n):
            win = torch.mean(x[j : j + n, i, :], 0, keepdim=True)  # 1xN
            sliders.append(win)
        sliders = torch.cat(sliders, 0)
        out[i] = torch.max(sliders, 0)[0]
    return out


def pool(out, lengths: torch.Tensor, pooling: str):
    if params.pooling == "mean":
        out = mean_pool(out, lengths)
    elif params.pooling == "max":
        out = max_pool(out, lengths)
    elif params.pooling == "min":
        out = min_pool(out, lengths)
    elif params.pooling == "hier":
        out = hier_pool(out, lengths)
    elif params.pooling == "sum":
        out = sum_pool(out, lengths)
    else:
        raise ValueError("No valid pooling operation specified!")
    return out
