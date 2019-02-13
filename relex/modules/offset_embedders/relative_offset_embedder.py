import torch
from allennlp.nn import util
from relex.modules.offset_embedders import OffsetEmbedder


@OffsetEmbedder.register("relative")
class RelativeOffsetEmbedder(OffsetEmbedder):
    def __init__(self, n_position: int, embedding_dim: int) -> None:
        super(RelativeOffsetEmbedder, self).__init__()

        self._n_position = n_position
        self._embedding_dim = embedding_dim
        self._embedding = torch.nn.Embedding(
            2 * n_position + 1, embedding_dim, padding_idx=0
        )
        self._embedding.weight.data[self._embedding.padding_idx].fill_(0)

    def get_output_dim(self) -> int:
        return self._embedding_dim

    def is_additive(self) -> bool:
        return False

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor, span: torch.Tensor
    ) -> torch.Tensor:
        # input -> [B x seq_len x d], offset -> [B x 2]
        batch_size, seq_len, _ = inputs.size()

        offset = span[:, 0]
        position_range = util.get_range_vector(
            seq_len, util.get_device_of(inputs)
        ).repeat((batch_size, 1))

        relative_positions = (
            1 + self._n_position + position_range - offset.unsqueeze(dim=1)
        )

        # mask padding so it won't receive a positional embedding
        relative_positions = relative_positions * mask.long()

        return self._embedding(relative_positions)
