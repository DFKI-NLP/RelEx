import torch
from allennlp.nn import util
from relex.modules.offset_embedders import OffsetEmbedder


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
    )  # apply sin on 0th,2nd,4th...embedding_dim
    position_enc[1:, 1::2] = np.cos(
        position_enc[1:, 1::2]
    )  # apply cos on 1st,3rd,5th...embedding_dim
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


@OffsetEmbedder.register("sine")
class SineOffsetEmbedder(OffsetEmbedder):
    def __init__(self, n_position: int, embedding_dim: int) -> None:
        super(SineOffsetEmbedder, self).__init__()

        self._n_position = n_position
        self._embedding_dim = embedding_dim
        self._embedding = torch.nn.Embedding(2 * n_position, embedding_dim)
        self._embedding.weight.data = position_encoding_init(
            2 * n_position, embedding_dim
        )

    def get_output_dim(self) -> int:
        return self._embedding_dim

    def is_additive(self) -> bool:
        return True

    def forward(self, inputs: torch.Tensor, span: torch.Tensor) -> torch.Tensor:
        # input-> [B x seq_len x d], offset -> [B x 2]
        batch_size, seq_len, _ = inputs.size()

        offset = span[:, 0]
        position_range = util.get_range_vector(
            seq_len, util.get_device_of(inputs)
        ).repeat((batch_size, 1))

        relative_positions = position_range + self._n_position - offset.unsqueeze(dim=1)

        return self._embedding(relative_positions)
