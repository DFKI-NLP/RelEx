import torch

from allennlp.common import Registrable


class OffsetEmbedder(torch.nn.Module, Registrable):
    """
    """

    default_implementation = "relative"

    def get_output_dim(self) -> int:
        """
        Returns the final output dimension that this ``OffsetEmbedder`` uses to represent each
        offset.  This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError

    def is_additive(self) -> bool:
        raise NotImplementedError
