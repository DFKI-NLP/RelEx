import torch


class WordDropout(torch.nn.Module):
    """
    Implementation of word dropout. Randomly drops out entire words (or characters) 
    in embedding space.
    """

    def __init__(self, p=0.05, fill_idx=1):
        super(WordDropout, self).__init__()
        self.p = p
        self.fill_idx = fill_idx

    def forward(self, x, mask):
        if not self.training or not self.p:
            return x

        m = x.data.new(1, x.size(1)).bernoulli_(self.p)
        dropout_mask = torch.autograd.Variable(m, requires_grad=False)
        dropout_mask = dropout_mask * mask
        dropout_mask = dropout_mask.expand_as(x)
        return x.masked_fill_(dropout_mask.byte(), self.fill_idx)
