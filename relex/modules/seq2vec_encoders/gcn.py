# brazenly stolen from: https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
# and https://github.com/tkipf/pygcn/blob/master/pygcn/models.py

import math
import torch
from overrides import overrides
from torch.nn.parameter import Parameter
from allennlp.nn import Activation
from allennlp.nn import util
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from relex.modules.seq2vec_encoders.utils import pool


class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # print("x:", x.shape)
        # print("adjacency:", adjacency.shape)
        # print("weight:", self.weight.shape)
        batch_size, seq_len, _ = x.size()

        x = x.view(batch_size * seq_len, -1)

        support = torch.mm(x, self.weight)
        support = support.view(batch_size, seq_len, -1)
        # output = torch.spmm(adjacency, support)
        output = torch.bmm(adjacency, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


@Seq2VecEncoder.register("gcn")
class GCN(Seq2VecEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0,
        gcn_layer_activation: Activation = None,
        pooling: str = "max",
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._activation = gcn_layer_activation or Activation.by_name("relu")()
        self._pooling = pooling

        self._gcn_layers = []

        gcn_input_size = input_size
        for layer_idx in range(num_layers):
            gcn_layer = GraphConvolution(
                in_features=gcn_input_size, out_features=hidden_size
            )
            self.add_module(f"gcn_layer_{layer_idx}", gcn_layer)
            self._gcn_layers.append(gcn_layer)
            gcn_input_size = hidden_size

        self.dropout = torch.nn.Dropout(dropout)

    @overrides
    def get_input_dim(self) -> int:
        return self.input_size

    @overrides
    def get_output_dim(self) -> int:
        return self.hidden_size * 3

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        head: torch.Tensor,
        tail: torch.Tensor,
        mask: torch.Tensor,
    ):
        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x * mask.float()

        output = x
        for i in range(len(self._gcn_layers)):
            gcn_layer_i = getattr(self, f"gcn_layer_{i}")
            output = self.dropout(self._activation(gcn_layer_i(output, adjacency)))

        batch_size, seq_len, _ = x.size()

        pos_range = util.get_range_vector(seq_len, util.get_device_of(x)).repeat(
            (batch_size, 1)
        )

        head_start = head[:, 0].unsqueeze(dim=1)
        head_end = head[:, 1].unsqueeze(dim=1)
        tail_start = tail[:, 0].unsqueeze(dim=1)
        tail_end = tail[:, 1].unsqueeze(dim=1)

        head_mask = (
            (torch.ge(pos_range, head_start) * torch.le(pos_range, head_end))
            .unsqueeze(-1)
            .long()
        )
        tail_mask = (
            (torch.ge(pos_range, tail_start) * torch.le(pos_range, tail_end))
            .unsqueeze(-1)
            .long()
        )

        pooled = []
        for m in [mask, head_mask, tail_mask]:
            pooled.append(
                pool(output, m, dim=1, pooling=self._pooling, is_bidirectional=False)
            )

        return torch.cat(pooled, dim=-1)
