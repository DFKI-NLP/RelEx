import torch


class WordDropout(torch.nn.Module):
    """
    Implementation of word dropout. Randomly drops out entire words (or characters)
    in embedding space.
    """

    def __init__(self, p: float = 0.05, fill_idx: int = 1):
        super(WordDropout, self).__init__()
        self.prob = p
        self.fill_idx = fill_idx

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        if not self.training or not self.prob:
            return inputs

        dropout_mask = inputs.data.new(1, inputs.size(1)).bernoulli_(self.prob)
        dropout_mask = torch.autograd.Variable(dropout_mask, requires_grad=False)
        dropout_mask = dropout_mask * mask
        dropout_mask = dropout_mask.expand_as(inputs)
        return inputs.masked_fill_(dropout_mask.byte(), self.fill_idx)
