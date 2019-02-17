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
        torch.nn.init.xavier_uniform_(self._embedding.weight.data)
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

        pos_range = util.get_range_vector(seq_len, util.get_device_of(inputs)).repeat(
            (batch_size, 1)
        )

        start_offset = span[:, 0].unsqueeze(dim=1)
        end_offset = span[:, 1].unsqueeze(dim=1)

        left_mask = torch.lt(pos_range, start_offset).long()
        middle_mask = (
            torch.ge(pos_range, start_offset) * torch.le(pos_range, end_offset)
        ).long()
        right_mask = torch.gt(pos_range, end_offset).long()

        offsets = start_offset * left_mask + end_offset * right_mask

        relative_positions = (
            1 + self._n_position + (pos_range - offsets) * (1 - middle_mask)
        )

        # mask padding so it won't receive a positional embedding
        relative_positions = relative_positions * mask.long()

        return self._embedding(relative_positions)
