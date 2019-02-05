import torch
from overrides import overrides
from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from relex.modules.seq2vec_encoders.utils import pool


@Seq2VecEncoder.register("seq2seq_pool")
class Seq2SeqPoolEncoder(Seq2VecEncoder):
    def __init__(self, encoder: Seq2SeqEncoder, pooling: str = "max") -> None:
        super().__init__()

        self._encoder = encoder
        self._pooling = pooling

    @overrides
    def get_input_dim(self) -> int:
        return self._encoder.get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self._encoder.get_output_dim()

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ):  # pylint: disable=arguments-differ
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        tokens = self._encoder(tokens, mask)
        return pool(tokens, mask.unsqueeze(-1), dim=1, pooling=self._pooling)
