import math
from typing import List

import torch

from allennlp.modules import Seq2VecEncoder
from allennlp.modules.matrix_attention import LinearMatrixAttention
from allennlp.nn import Activation
from allennlp.nn.util import masked_softmax, weighted_sum
from overrides import overrides
from torch import nn
from torch.nn import Parameter

from relex.modules.seq2vec_encoders.utils import PoolingScope, scoped_pool


@Seq2VecEncoder.register('gat')
class GAT(Seq2VecEncoder):

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int = 1,
            num_heads: int = 1,
            activation: Activation = None,
            input_dropout: float = 0.0,
            att_dropout: float = 0.0,
            pooling: str = "max",
            pooling_scope: List[str] = None,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._pooling = pooling

        layer_activation = activation or Activation.by_name("leaky_relu")(0.2)

        self._gat_layers = []
        for layer_idx in range(num_layers):
            gat_layer = GraphAttentionLayer(
                input_dim=input_dim if layer_idx == 0 else hidden_dim,
                output_dim=hidden_dim,
                num_heads=num_heads,
                input_dropout=input_dropout,
                att_dropout=att_dropout,
                activation=layer_activation
            )
            self._gat_layers.append(gat_layer)
            self.add_module(f"gat_layer_{layer_idx}", gat_layer)

        if pooling_scope is None:
            self._pooling_scope = [PoolingScope.SEQUENCE,
                                   PoolingScope.HEAD,
                                   PoolingScope.TAIL]
        else:
            self._pooling_scope = [PoolingScope(scope.lower())
                                   for scope in pooling_scope]

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._hidden_dim * 3

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            head: torch.Tensor,
            tail: torch.Tensor,
            adjacency: torch.Tensor
    ):
        output = x
        for gat_layer in self._gat_layers:
            output = gat_layer(output, mask, adjacency)

        return scoped_pool(
            output,
            mask,
            pooling=self._pooling,
            pooling_scopes=self._pooling_scope,
            is_bidirectional=False,
            head=head,
            tail=tail
        )


class GraphAttentionLayer(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_heads: int = 1,
                 activation: Activation = None,
                 input_dropout: float = 0.0,
                 att_dropout: float = 0.0) -> None:
        super().__init__()
        self._hidden_dim = output_dim
        self._weight_vector = Parameter(torch.FloatTensor(input_dim, self._hidden_dim))
        if activation is not None:
            self.activation = activation
        else:
            self.activation = lambda x: x

        attention_dim = output_dim / num_heads
        assert attention_dim.is_integer(), "output dim must be divisible by number of heads"
        self._attention_dim = int(attention_dim)
        self._num_heads = num_heads
        self.matrix_attention = LinearMatrixAttention(tensor_1_dim=self._attention_dim,
                                                      tensor_2_dim=self._attention_dim,
                                                      combination='x,y')

        if input_dropout is not None and input_dropout > 0:
            self.input_dropout = nn.Dropout(input_dropout)
        else:
            self.input_dropout = lambda x: x

        if att_dropout is not None and att_dropout > 0:
            self.att_dropout = nn.Dropout(att_dropout)
        else:
            self.att_dropout = lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self._weight_vector.size(1))
        self._weight_vector.data.uniform_(-stdv, stdv)

    def masked_self_attention(self, x, mask, adjacency):
        batch_size, seq_len, _ = x.size()

        # shape (num_heads * batch_size, seq_len, attention_dim)
        x = x.view(batch_size, seq_len, self._num_heads, self._attention_dim)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size * self._num_heads, seq_len, self._attention_dim)

        # shape (num_heads * batch_size, seq_len, seq_len)
        adjacency_per_head = adjacency \
            .unsqueeze(1) \
            .repeat(1, self._num_heads, 1, 1) \
            .view(batch_size * self._num_heads, seq_len, seq_len).byte()

        # shape (num_heads * batch_size, seq_len, seq_len)
        mask_per_head = mask.repeat(1, self._num_heads).view(batch_size * self._num_heads, seq_len).float()
        mask_per_head = mask_per_head.unsqueeze(2)
        mask_per_head = mask_per_head.bmm(mask_per_head.transpose(1,2)).byte()

        # Only attend on nodes visible in the adjacency matrix
        attention_mask = adjacency_per_head & mask_per_head
        attention_mask = self.att_dropout(attention_mask)

        similarities = self.matrix_attention(x, x)

        # shape (num_heads * batch_size, seq_len, seq_len)
        # Normalise the distributions, using the same mask for all heads.
        attention = masked_softmax(similarities, attention_mask, memory_efficient=True)

        # Take a weighted sum of the values with respect to the attention
        # distributions for each element in the num_heads * batch_size dimension.
        # shape (num_heads * batch_size, seq_len, attention_dim)
        outputs = weighted_sum(x, attention)

        # Reshape back to original shape (batch_size, timesteps, hidden_dim)

        # shape (batch_size, num_heads, timesteps, values_dim/num_heads)
        outputs = outputs.view(batch_size, self._num_heads, seq_len, self._attention_dim)
        # shape (batch_size, seq_len, num_heads, values_dim/num_heads)
        outputs = outputs.transpose(1, 2).contiguous()
        # shape (batch_size, seq_len, hidden_dim)
        outputs = outputs.view(batch_size, seq_len, self._hidden_dim)

        return outputs

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            adjacency: torch.Tensor
    ):
        batch_size, seq_len, _ = x.size()

        x = self.input_dropout(x)
        x = x.view(batch_size * seq_len, -1)
        x = torch.mm(x, self._weight_vector)
        x = x.view(batch_size, seq_len, -1)
        x = self.masked_self_attention(x, mask, adjacency)
        x = self.activation(x)

        return x

