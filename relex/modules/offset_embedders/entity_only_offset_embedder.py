import torch
from allennlp.nn import util
from relex.modules.offset_embedders import OffsetEmbedder


@OffsetEmbedder.register("entity_only")
class EntityOnlyOffsetEmbedder(OffsetEmbedder):
    def __init__(self, n_position: int, embedding_dim: int) -> None:
        super(EntityOnlyOffsetEmbedder, self).__init__()

        self._n_position = n_position
        self._embedding_dim = embedding_dim

    def get_output_dim(self) -> int:
        return self._embedding_dim

    def is_additive(self) -> bool:
        return False

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor, span: torch.Tensor
    ) -> torch.Tensor:
        # input -> [B x seq_len x d], offset -> [B x 2]
        batch_size, seq_len, _ = inputs.size()

        offset = span[:, 0].unsqueeze(-1)
        position_range = util.get_range_vector(
            seq_len, util.get_device_of(inputs)
        ).repeat((batch_size, 1))

        offset_mask = position_range == offset

        position_markers = inputs.new_ones((batch_size, seq_len), requires_grad=True)
        position_markers = position_markers * offset_mask.float()
        position_markers = position_markers.unsqueeze(-1)

        return position_markers
