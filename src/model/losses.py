import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """The Dice loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.
        Returns:
            loss (torch.Tensor) (0): The dice loss.
        """
        # Get the one-hot encoding of the ground truth label.
        try:
            target = torch.zeros_like(output).scatter_(1, target, 1)
        except RuntimeError:
            target = torch.zeros_like(tmp)
            output = torch.zeros_like(output)
        else:

        # Calculate the dice loss.
            reduced_dims = list(range(2, output.dim())) # (N, C, *) --> (N, C)
            intersection = 2.0 * (output * target).sum(reduced_dims)
            union = (output ** 2).sum(reduced_dims) + (target ** 2).sum(reduced_dims)
            score = intersection / (union + 1e-10)
        return 1 - score.mean()
