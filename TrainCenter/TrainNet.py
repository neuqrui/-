import time
from typing import Any
import numpy as np
from utils.bar import Bar
import torch

import torch.optim as optim
from utils.AverageMeter import AverageMeter
import torch.nn as nn
from game_rules.dotsboxes import Game

class TrainNet:
    """
    Neural network training class
    """

    def __init__(self, args, net_work, memory):
        self.memory = memory
        self.args = args
        self.game = Game(self.args)
        self.board_size = args.board_size
        self.net_work = net_work
        self.net_work_parallel = None
        self.loss_mse = nn.MSELoss(reduction='sum')
        self.reg_loss = Regularization(self.args, self.net_work, self.args.weight_decay, p=2).to(self.args.device)
        self.init_weight()

        if self.args.cuda:
            self.net_work.cuda()
            if self.args.gpu_parallel:
                self.net_work_parallel = nn.DataParallel(self.net_work, device_ids=self.args.gpu_ids).cuda()
                self.net_work_parallel.to(self.args.device)

    def train(self, replay_buffer, num_iter):
        """
        examples: list of examples, each example is of form (board, pi, v)
        :param replay_buffer: (object) Experience RePlay Pool Object
        :param num_iter: (int): num of iteration
        """
        if num_iter > self.args.lr_iter_threshold:  # 80
            lr = self.args.lr / 4
        else:
            lr = self.args.lr
        self.net_work.train()
        if self.args.cuda and self.args.gpu_parallel:
            optimizer = optim.Adam(self.net_work_parallel.parameters(), lr=lr, weight_decay=self.args.weight_decay)
        else:
            optimizer = optim.Adam(self.net_work.parameters(), lr=lr, weight_decay=self.args.weight_decay)
        pi_losses_sum = 0
        v_losses_sum = 0
        reg_losses_sum = 0
        data_num = replay_buffer.get_total_data_num()

        for epoch in range(self.args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            data_time = AverageMeter()
            batch_time = AverageMeter()
            reg_losses = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            start = time.time()

            bar = Bar('Training', max=int(data_num / self.args.batch_size))
            batch_idx = 0
            while batch_idx < int(data_num / self.args.batch_size):
                data, target_policy, target_values, iter_num, num_lay = replay_buffer.get_batch()
                new_data = []
                for key in data[0]:
                    new_data.append(self.network_state(key))
                new_data = torch.tensor(np.array(new_data).astype('float64')).float()

                l_pi, l_v, l_reg = torch.tensor(0, dtype=torch.float), torch.tensor(0, dtype=torch.float), torch.tensor(
                    0, dtype=torch.float)
                if self.args.cuda:
                    l_pi, l_v, l_reg = l_pi.contiguous().cuda(), l_v.contiguous().cuda(), l_reg.contiguous().cuda()
                    # data, target_policy, target_values, iter_num = data.contiguous().cuda(), target_policy.contiguous().cuda(), target_values.contiguous().cuda(), iter_num.contiguous().cuda()
                    for i in range(self.args.num_net):
                        new_data, target_policy[i], target_values[i], iter_num[i] = new_data.contiguous().cuda(), \
                        target_policy[i].contiguous().cuda(), target_values[i].contiguous().cuda(), iter_num[
                            i].contiguous().cuda()
                data_time.update(time.time() - start)
                # gpu parallel train
                out_pi, out_v = self.net_work_parallel(
                    data) if self.args.cuda and self.args.gpu_parallel else self.net_work(new_data)
                l_reg += self.reg_loss(
                    self.net_work_parallel) if self.args.cuda and self.args.gpu_parallel else self.reg_loss(
                    self.net_work)
                for i in range(self.args.num_net):
                    iter_np = torch.tensor(np.array([num_iter for _ in range(iter_num[i].shape[0])]))
                    # 数据越久远，权重占比越小
                    iter_deca = torch.pow(self.args.replay_decay_rate, (iter_np.contiguous().cuda() - iter_num[
                        i])).contiguous().cuda() if self.args.cuda else torch.pow(self.args.replay_decay_rate,
                                                                                  (iter_np - iter_num[i]))
                    l_pi += self.loss_pi(target_policy[i], out_pi, iter_deca)
                    l_v += self.loss_v(target_values[i], out_v, iter_deca)
                total_loss = self.args.C_pi_loss * l_pi + self.args.C_v_loss * l_v + self.args.C_reg_loss * l_reg
                # record loss
                pi_losses.update(l_pi.item(), self.args.batch_size)  # data_num是replay_buffer中总的数据量，应该用batch_size吧
                v_losses.update(l_v.item(), self.args.batch_size)
                reg_losses.update(l_reg.item(), self.args.batch_size)
                # compute gradient and do Adam step
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.net_work.parameters(), 3, norm_type=2)
                optimizer.step()
                # measure elapsed time
                batch_time.update(time.time() - start)
                start = time.time()
                batch_idx += 1
                # plot progress
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.2f}s | Total: {total:} | ETA: {eta:} | L_pi: {lpi:.3f} | L_v: {lv:.3f} | L_reg: {lreg:.3f}'.format(
                    batch=batch_idx,
                    size=int(data_num / self.args.batch_size),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    lpi=pi_losses.avg,
                    lv=v_losses.avg,
                    lreg=reg_losses.avg,
                )
                bar.next()
            pi_losses_sum += pi_losses.avg
            v_losses_sum += v_losses.avg
            reg_losses_sum += reg_losses.avg
            bar.finish()
        if self.args.cuda and self.args.gpu_parallel:
            self.net_work.load_state_dict(self.net_work_parallel.state_dict(), False)
        self.memory.save_loss(pi_losses_sum / self.args.epochs, v_losses_sum / self.args.epochs,
                              reg_losses_sum / self.args.epochs)

    def network_state(self, board):
        """
        神经网络棋盘
        状态形式：2 * 宽 * 高
        """
        board = np.array(board).reshape(11, 11)
        red_territory = self.game.get_number_of_territory(board, self.game.RED)
        blue_territory = self.game.get_number_of_territory(board, self.game.BLUE)
        v = red_territory - blue_territory
        match = {(0, 1): 1, (0, 2): 2, (0, 3): 3, (0, 4): 4, (1, 0): 5, (1, 1): 6, (1, 2): 7, (1, 3): 8, (1, 4): 9, (2, 0): 10, (2, 1): 11, (2, 2): 12, (2, 3): 13, (2, 4): 14, (3, 0): 15, (3, 1): 16, (3, 2): 17, (3, 3): 18, (3, 4): 19, (4, 0): 20, (4, 1): 21, (4, 2): 22, (4, 3): 23, (4, 4): 24}
        if self.args.input_channels == 2:
            # 使用2个11x11的二值特征平面来描述当前的局面
            # 第一个平面表示当前棋盘上的边，有边的位置是1，没边的位置是0
            # 第二个平面表示红方-蓝方格子的差值v, 那么棋盘所有位置都是v
            state = np.zeros((2, self.board_size, self.board_size))
            state[0][board == -2] = 1
            state[0][board == 2] = 1
            state[1][:, :] = v / 25

        elif self.args.input_channels == 26:
            board = np.array(board).reshape(11, 11)
            state = np.zeros((26, self.board_size, self.board_size))
            # 第0个channel，已经存在的边
            state[0][board == -2] = 1
            state[0][board == 2] = 1
            for key in self.game.all_sides_2D:
                if board[key[0]][key[1]] == 0:
                    a, b = self.judge_side_type(key[0], key[1], board)
                    state[match[(a, b)]][key[0]][key[1]] = 1
            state[25][:, :] = (v + 25) / 50

        return state

    def judge_side_type(self, x, y, board):
        assert (x + y) % 2 == 1
        # 竖边
        sx = [1, 0, -1, -1, 0, 1]
        sy = [-1, -2, -1, 1, 2, 1]
        # 横边
        hx = [-1, -2, -1, 1, 2, 1]
        hy = [-1, 0, 1, 1, 0, -1]
        board_temp = board.copy()
        board_temp = board_temp.reshape(self.board_size, self.board_size)
        state = 1
        if x % 2 == 0:  # 横边
            first_cnt, second_cnt = state, state
            if x == 0:  # 上边界
                for k in range(3, 6):
                    nx = x + hx[k]
                    ny = y + hy[k]
                    if board_temp[nx][ny] != 0:
                        second_cnt += 1
                return 0, second_cnt
            elif x == self.board_size - 1:  # 下边界
                for k in range(3):
                    nx = x + hx[k]
                    ny = y + hy[k]
                    if board_temp[nx][ny] != 0:
                        first_cnt += 1
                return first_cnt, 0
            else:
                for k in range(3):
                    nx = x + hx[k]
                    ny = y + hy[k]
                    if board_temp[nx][ny] != 0:
                        first_cnt += 1
                for k in range(3, 6):
                    nx = x + hx[k]
                    ny = y + hy[k]
                    if board_temp[nx][ny] != 0:
                        second_cnt += 1
                return first_cnt, second_cnt
        if y % 2 == 0:  # 竖边
            first_cnt, second_cnt = state, state
            if y == 0:
                for k in range(3, 6):
                    nx = x + sx[k]
                    ny = y + sy[k]
                    if board_temp[nx][ny] != 0:
                        second_cnt += 1
                return 0, second_cnt
            elif y == self.board_size - 1:
                for k in range(3):
                    nx = x + sx[k]
                    ny = y + sy[k]
                    if board_temp[nx][ny] != 0:
                        first_cnt += 1
                return first_cnt, 0
            else:
                for k in range(3):
                    nx = x + sx[k]
                    ny = y + sy[k]
                    if board_temp[nx][ny] != 0:
                        first_cnt += 1
                for k in range(3, 6):
                    nx = x + sx[k]
                    ny = y + sy[k]
                    if board_temp[nx][ny] != 0:
                        second_cnt += 1
                return first_cnt, second_cnt

    def calculate_step_loss(self, replay_buffer):
        """
        Calculate the policy cross entropy and value mean square error loss of each step
        :param replay_buffer:(object) experience replay pool object
        """
        step_loss_p = [0 for _ in range(self.args.num_max_layers)]
        num_layers = [0 for _ in range(self.args.num_max_layers)]
        step_loss_v = [0 for _ in range(self.args.num_max_layers)]
        for layers, _, net_pi, pi, net_v, v in replay_buffer.trajectory:
            step_loss_p[layers] = (step_loss_p[layers] * num_layers[layers] + self.loss_pi(torch.tensor(pi, dtype=torch.float64), torch.tensor(net_pi, dtype=torch.float64), torch.tensor([1], dtype=torch.float64)).numpy().tolist() * self.board_size ** 2) / (num_layers[layers] + 1)
            step_loss_v[layers] = (step_loss_v[layers] * num_layers[layers] + self.loss_v(torch.tensor(net_v, dtype=torch.float64), torch.tensor(v, dtype=torch.float64),  torch.tensor([1], dtype=torch.float64)).numpy().tolist()) / (num_layers[layers] + 1)
            num_layers[layers] += 1
        return step_loss_p, step_loss_v

    def predict(self, board, net_work):
        """
        Network prediction function

        :param board:(numpy) the current board
        :param net_work:(object) network objects
        :returns pi:(list) prediction policy
                 v:(int) prediction value
        """
        board = self.network_state(board)
        board = torch.tensor(board.astype('float64')).float()
        if self.args.cuda:
            board = board.contiguous().cuda()
        net_work.eval()
        with torch.no_grad():
            pi, v = net_work(board.view(-1, self.args.input_channels, self.board_size, self.board_size))
        return pi.cpu().numpy()[0], v.cpu().numpy()[0]

    def loss_pi(self, labels, outputs, iter_deca):
        """
        Computing policy cross entropy loss
        :param labels:(torch.tensor) target labels
        :param outputs:(torch.tensor) network output
        :param iter_deca:(torch.tensor)
        :return loss_pi:(float) the loss
        """
        return self.cross_entropy_loss(labels, outputs, iter_deca)

    def loss_v(self, labels, outputs, iter_deca):
        """
        Calculate value network loss
        :param labels:(torch.tensor) supervisory signal
        :param outputs:(torch.tensor) neural network output
        :param iter_deca:(torch.tensor)
        :return: loss:(float) loss value
        """
        # return self.loss_mse(outputs.view(-1), labels) / labels.size()[0]
        mse_loss_fn = nn.MSELoss(reduction='none')
        mse_loss = mse_loss_fn(outputs.view(-1), labels) / labels.size()[0]
        weighted_loss = torch.sum(mse_loss*iter_deca)/labels.size()[0]
        # return torch.sum(torch.pow((outputs.view(-1) - labels), 2) * iter_deca) / labels.size()[0]
        return weighted_loss

    @staticmethod
    def cross_entropy_loss(labels, net_outs, iter_deca, epsilon=1e-10):
        """
        Cross entropy loss function
        :param labels:(torch.tensor)  target labels
        :param net_outs:(torch.tensor)  network output
        :param iter_deca:(torch.tensor)
        :return loss:(float) loss value
        """
        assert net_outs.shape == labels.shape
        net_outs = torch.clamp(net_outs, min=epsilon)
        if len(list(labels.size())) == 2:
            sum_loss = torch.sum(torch.sum(-labels * torch.log(net_outs), dim=1) * iter_deca)
        else:
            sum_loss = torch.sum(-labels * torch.log(net_outs) * iter_deca)
        return sum_loss / labels.shape[0]

    def init_weight(self):
        """
        Initialize network parameters
        """
        for m in self.net_work.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Regularization(torch.nn.Module):
    """
    Regular term class
    """

    def __init__(self, args, model, weight_decay, p=2):
        """
        :param model:(object) network model
        :param weight_decay: Regularization parameters
        :param p:(int 1/2) The power index value in norm calculation is 2 norm by default,When p = 0 is L2 regularization, P = 1 is L1 regularization
        """
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.device = args.device
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        # self.weight_info(self.weight_list)

    def to(self, device):
        """
        Specify operation mode
        :param device: cuda or cpu
        :return:
        """
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        """
        Forward calculation
        :param model:(object) network model
        :return:
        """
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    @staticmethod
    def get_weight(model):
        """
        Get the weight list of the model
        :param model:(object) network model
        :return:
        """
        weight_list = []
        for name, param in model.named_parameters():
            weight = (name, param)
            weight_list.append(weight)
        return weight_list

    @staticmethod
    def regularization_loss(weight_list, weight_decay, p=2):
        """
        Calculation of tensor norm and regularization loss
        :param weight_list: the same as above
        :param p:(int 1/2) the same as above
        :param weight_decay: lamda parameter in regularization loss
        :return:
        """
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss += l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    @staticmethod
    def weight_info(weight_list):
        """
        Print weight list information
        :param weight_list: the same as above
        :return:
        """
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")

    def _forward_unimplemented(self, *inputs: Any) -> None:
        pass
