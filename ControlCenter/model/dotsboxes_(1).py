from typing import Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.abstract_model import AbstractNet, ResidualBlock

# 定义一个带参数的神经网络自定义层
class _Fusion_Layer(nn.Module):
    def __init__(self, input_dim: int, board_x: int, board_y: int, output_dim: int,
                 # device = 'cpu', dtype = 'float32'
                 ) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        super(_Fusion_Layer, self).__init__()

        # 定义一个融合层
        self.input_dim = input_dim
        self.board_x = board_x
        self.board_y = board_y
        self.output_dim = output_dim

        # 定义一个权重矩阵和偏置向量
        self.weight = nn.Parameter(
            torch.Tensor(self.input_dim, self.board_x, self.board_y, self.output_dim)
        )
        self.bias = nn.Parameter(
            torch.Tensor(self.output_dim, self.board_x, self.board_y)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.input_dim
        bound = 1 / (math.sqrt(fan_in) + 1e-9)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_dim, self.board_x, self.board_y)
        x = torch.einsum('ixyo, bixy -> boxy', [self.weight, x]) + self.bias
        return x

class NetWork(AbstractNet):
    """
    Neural network model class object
    """

    def __init__(self, args) -> None:
        """
        :param args:(object) class object of super parameter
        """
        super().__init__()
        self.args = args
        self.input_channel = self.args.input_channels
        self.board_x = self.args.board_size
        self.board_y = self.args.board_size

        self.common_block = nn.Sequential(
            ResidualBlock(self.input_channel, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 96),
            ResidualBlock(96, 96),
            ResidualBlock(96, 96),
        )

        self.policy_block = nn.Sequential(
            ResidualBlock(96, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),

            _Fusion_Layer(64, self.board_x, self.board_x, 1),   # BATCH_SIZE, 1, BOARD_X, BOARD_Y
            nn.BatchNorm2d(1),
        )

        self.value_block = nn.Sequential(
            ResidualBlock(96, 64),
            ResidualBlock(64, 64),

            _Fusion_Layer(64, self.board_x, self.board_x, 1),
            nn.BatchNorm2d(1),
            nn.Tanh(),
        )

        self.value = nn.Sequential(
            nn.Linear(self.board_x * self.board_y, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.BatchNorm1d(1),
            # nn.Tanh(),
        )


    def forward(self, data):
        """
        Forward propagation function
        :param data:(torch.tensor) Input network data (n, 2, x, y)
        :returns pi:(torch.tensor) policy output
                 v:(torch.tensor) value output
        """

        assert data.dtype == torch.float32

        # reshape input data
        data = data.reshape(-1, self.input_channel, self.board_x, self.board_y)

        # common block
        x = self.common_block(data)

        policy = self.my_softmax(self.policy_block(x).view(-1, self.board_x * self.board_y))

        value = self.value(self.value_block(x).view(-1, self.board_x * self.board_y))

        return policy, value

    def my_softmax(self, x):
        max_x = x.max(dim=-1).values.unsqueeze(-1)
        return (x - max_x + 1e-9).softmax(dim=-1)

    def _forward_unimplemented(self, *inputs: Any) -> None:
        pass
