from typing import List

import torch
from overrides import overrides
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from relex.modules.seq2vec_encoders.utils import PoolingScope, scoped_pool


@Seq2VecEncoder.register("seq2seq_pool")
class Seq2SeqPoolEncoder(Seq2VecEncoder):
    def __init__(self,
                 encoder: Seq2SeqEncoder,
                 pooling: str = "max",
                 pooling_scope: List[str] = None) -> None:
        super().__init__()

        self._encoder = encoder
        self._pooling = pooling
        if pooling_scope is None:
            self._pooling_scope = [PoolingScope.SEQUENCE]
        else:
            self._pooling_scope = [PoolingScope(scope)
                                   for scope in pooling_scope]

    @overrides
    def get_input_dim(self) -> int:
        return self._encoder.get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self._encoder.get_output_dim()

    def forward(
            self,
            tokens: torch.Tensor,
            mask: torch.Tensor,
            head: torch.LongTensor = None,
            tail: torch.LongTensor = None
    ):  # pylint: disable=arguments-differ
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        tokens = self._encoder(tokens, mask)

        return scoped_pool(
            tokens,
            mask,
            pooling=self._pooling,
            pooling_scopes=self._pooling_scope,
            is_bidirectional=self._encoder.is_bidirectional(),
            head=head,
            tail=tail
        )
