from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as f


class AbstractNet(ABC, nn.Module):

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, in_data):
        pass

    @abstractmethod
    def _forward_unimplemented(self, *inputs: Any) -> None:
        pass


class ResidualBlock(nn.Module):
    """

    """
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel))
        self.right = nn.Sequential(
            # Use Conv2d with the kernel_size of 1, without padding to improve the parameters of the network
            nn.Conv2d(in_channel, out_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel))
        if torch.cuda.is_available():
            self.left.cuda()
            self.right.cuda()

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out = out + residual
        return f.relu(out)

    def _forward_unimplemented(self, *inputs: Any) -> None:
        pass
