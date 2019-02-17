from typing import Sequence, Dict, List, Callable, Optional

import torch
import numpy as np
from overrides import overrides
from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn.activations import Activation
from relex.modules.seq2vec_encoders.utils import pool


@Seq2VecEncoder.register("bag_of_embeddings")
class BagOfEmbeddings(Seq2VecEncoder):
    """
    """

    def __init__(
        self,
        embedding_dim: int,
        pooling: str = "sum",
        projection_dim: Optional[int] = None,
        activation: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._embedding_dim = embedding_dim
        self._pooling = pooling
        self._projection_dim = projection_dim

        self._activation = Activation.by_name(activation) if activation else None

        if self._projection_dim:
            self._projection = torch.nn.Linear(
                self._embedding_dim, self._projection_dim
            )

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._projection_dim or self._embedding_dim

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ):  # pylint: disable=arguments-differ
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()

        if self._projection_dim is not None:
            tokens = self._projection(tokens)

        out = pool(
            tokens,
            mask.unsqueeze(-1),
            dim=1,
            pooling=self._pooling,
            is_bidirectional=False,
        )

        if self._activation is not None:
            out = self._activation(out)

        return out
