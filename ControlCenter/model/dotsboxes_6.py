from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as f
from model.abstract_model import AbstractNet, ResidualBlock
from game_args.dotsboxes_6 import Args

class NetWork(AbstractNet):
    """
    Neural network model class object
    """

    def __init__(self, args):
        """
        :param args:(object) class object of super parameter
        """
        self.args = args
        self.inchannels = self.args.input_channels
        assert self.args.CenterName == "Control Center"
        super(NetWork, self).__init__()
        self.board_x, self.board_y = args.board_size, args.board_size
        self.layer1 = ResidualBlock(self.inchannels, 96)
        self.layer2 = ResidualBlock(96, 96)
        self.layer3 = ResidualBlock(96, 96)
        self.layer4 = ResidualBlock(96, 64)
        self.layer5 = ResidualBlock(64, 64)
        self.layer6 = ResidualBlock(64, 64)

        self.layer1_p = ResidualBlock(64, 64)
        self.layer2_p = ResidualBlock(64, 64)
        self.layer1_v = ResidualBlock(64, 64)
        self.layer2_v = ResidualBlock(64, 16)
        self.fc1_p = nn.Linear(64 * self.board_x * self.board_y, 256)
        self.fc_bn1_p = nn.BatchNorm1d(256)
        self.fc1_v = nn.Linear(16 * self.board_x * self.board_y, 64)
        self.fc_bn1_v = nn.BatchNorm1d(64)

        # policy network
        self.fc3 = nn.Linear(256, self.board_x * self.board_y)
        # value network
        self.fc4 = nn.Linear(64, 1)

    def forward(self, data):
        """
        Forward propagation function
        :param data:(torch.tensor) Input network data
        :returns pi:(torch.tensor) policy output
                 v:(torch.tensor) value output
        """
        data = data.reshape(-1, self.inchannels, self.board_x, self.board_y)
        # print(data[-1, 0, 0:self.board_x, 0:self.board_x].shape)
        x = self.layer1(data)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x1 = self.layer1_p(x)
        x_p = self.layer2_p(x1)
        x2 = self.layer1_v(x)
        x_v = self.layer2_v(x2)
        x_p = x_p.view(-1, 64 * self.board_x * self.board_y)
        x_v = x_v.view(-1, 16 * self.board_x * self.board_y)
        x_p = f.relu(self.fc_bn1_p(self.fc1_p(x_p)))
        x_v = f.relu(self.fc_bn1_v(self.fc1_v(x_v)))
        pi_temp = self.fc3(x_p)
        queen_pi = f.softmax(pi_temp, dim=1)
        v = self.fc4(x_v)
        return queen_pi, torch.tanh(v)

    def _forward_unimplemented(self, *inputs: Any) -> None:
        pass
