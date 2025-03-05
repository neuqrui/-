import importlib

# from ControlCenter.Storage import Logger
from TrainNet import TrainNet
from Storage import Log, Storage
from pathos import multiprocessing
import sys
from multiprocessing.managers import BaseManager
import numpy as np
import random


class QueueManager(BaseManager):
    pass


class SelfPlay:
    """

    """

    def __init__(self, game, Mcts, args):
        """
        :param Mcts:(class)
        :param game:(object)
        :param args:(object)
        """
        self.Control_center_IP = '127.0.0.1'   # '172.16.95.133'
        self.Control_center_Port = 10300   # 61188
        QueueManager.register('get_task_queue')
        QueueManager.register('get_result_queue')
        print('Connect to server %s...' % self.Control_center_IP)
        manager = QueueManager(address=(self.Control_center_IP, self.Control_center_Port), authkey=b'abc')
        manager.connect()
        global task
        task = manager.get_task_queue()
        global result
        result = manager.get_result_queue()
        self.game = game
        self.args = None
        self.num_RED_win = 0
        self.num_BLUE_win = 0
        self.num_iter = 1
        self.memory = None
        self.model_module = None
        self.trainer = None
        self.Mcts = Mcts
        self.mcts = None
        str_list = args.GAME_NAME.split('_')
        self.replay_buffer_module = importlib.import_module("replay_buffer." + str_list[0])
        self.replay_buffer = self.replay_buffer_module.ReplayBuffer(args)

    def run(self):
        """
        Generate board sequence through self-game_rules
        :return:
        """
        while True:
            try:
                self.num_RED_win = 0
                self.num_BLUE_win = 0
                # Hyper-parameter args is from the master riot
                model_data, self.args, num_iter = task.get(block=True, timeout=None)
                self.memory = Storage(self.game, self.args)
                self.model_module = importlib.import_module("model." + self.args.GAME_NAME)
                self.model_module.NetWork(self.args)
                self.trainer = TrainNet(self.args, self.model_module.NetWork(self.args), self.memory)
                self.mcts = self.Mcts(self.game, self.args, self.trainer, self.replay_buffer)
                self.args.num_params = sum(param.numel() for param in self.trainer.net_work.parameters())
                Log.print_args(self.args)
                Log.print_string("Load latest model ......")
                self.memory.save_model(model_data)
                model_weight = self.memory.load_model_weight("best")
                self.trainer.net_work.load_state_dict(model_weight)

                Log.print_iter_num(num_iter)
                data_buffer, trajectory, root_Q_total_num, root_Q_right_num, search_step_root_Q_total_num, search_step_root_Q_right_num = self.play_one_iter_game(num_iter)
                Log.print_vs_info(self.args, self.num_RED_win, self.num_BLUE_win)
                Log.print_str_int("Training data num", len(data_buffer))
                result.put((data_buffer, trajectory, self.num_RED_win, self.num_BLUE_win, root_Q_total_num, root_Q_right_num, search_step_root_Q_total_num, search_step_root_Q_right_num))
            except Exception as e:
                print('Error {} occurred probably because task queue is empty.'.format(e))

    def play_one_iter_game(self, num_iter):
        """
        Open multi-process to generate the data needed for one iteration
        :param num_iter:(int)
        :return:
        """
        re_data, re_trajectory = [], []
        total_data_num = 0
        root_Q_total_num, root_Q_right_num, search_step_root_Q_total_num, search_step_root_Q_right_num = 0, 0, 0, 0
        # 从头开始下
        for i in range(self.args.num_play_game):
            if len(re_data) > self.args.min_data_num:
                Log.print_string('The num of data reach the num: ' + str(self.args.min_data_num) + '; current game num: ' + str(i + 1) + '; Stop !')
                return re_data, re_trajectory
            Log.print_string('Play the game from init board')
            Log.print_play_num(i + 1)
            if self.args.multiprocess:
                # Multi process
                return_value = []
                pool = multiprocessing.Pool(processes=self.args.num_process)
                for proc in range(self.args.num_process):
                    return_value.append(
                        pool.apply_async(self.open_multiprocess, (None, None, None, proc, num_iter)))
                pool.close()
                pool.join()

                for proc in range(self.args.num_process):
                    num_RED_win, num_BLUE_win, trajectory = return_value[proc].get()
                    self.num_RED_win += num_RED_win
                    self.num_BLUE_win += num_BLUE_win
                    data = self.memory.load_process_data(proc)
                    re_data.extend(data)
                    re_trajectory.extend(trajectory)
            else:
                # Single process
                # print("computer center mct args: ", self.mcts.args.CenterName)
                loss_player, data, trajectory, chess_record, a, b, c, d = self.game.play_one_game_new(in_layer=None, in_board=None,
                                                                                      in_player=None,
                                                                                      proc_id=self.args.SingleProcess,
                                                                                      num_iter=num_iter, mcts=self.mcts)
                root_Q_right_num += b
                root_Q_total_num += a
                search_step_root_Q_right_num += d
                search_step_root_Q_total_num += c
                self.memory.save_chess_record(num_iter, chess_record)
                re_data.extend(data)
                re_trajectory.extend(trajectory)
                # 统计和棋数据在这里
                if loss_player == self.game.BLUE:
                    self.num_RED_win += 1
                elif loss_player == self.game.RED:
                    self.num_BLUE_win += 1

        return re_data, re_trajectory,  root_Q_total_num, root_Q_right_num, search_step_root_Q_total_num, search_step_root_Q_right_num

    # def play_one_iter_game(self, num_iter):
    #     """
    #     Open multi-process to generate the data needed for one iteration
    #     :param num_iter:(int)
    #     :return:
    #     """
    #     re_data, re_trajectory = [], []
    #     total_data_num = 0
    #     root_Q_total_num, root_Q_right_num, search_step_root_Q_total_num, search_step_root_Q_right_num = 0, 0, 0, 0
    #     # 从头开始下
    #     if self.args.start_search_step_threshold <= self.args.delta_search_step_threshold:
    #         for i in range(self.args.num_play_game):
    #             if len(re_data) > self.args.min_data_num:
    #                 Log.print_string('The num of data reach the num: ' + str(self.args.min_data_num) + '; current game num: ' + str(i + 1) + '; Stop !')
    #                 return re_data, re_trajectory
    #             Log.print_string('Play the game from init board')
    #             Log.print_play_num(i + 1)
    #             if self.args.multiprocess:
    #                 # Multi process
    #                 return_value = []
    #                 pool = multiprocessing.Pool(processes=self.args.num_process)
    #                 for proc in range(self.args.num_process):
    #                     return_value.append(
    #                         pool.apply_async(self.open_multiprocess, (None, None, None, proc, num_iter)))
    #                 pool.close()
    #                 pool.join()
    #
    #                 for proc in range(self.args.num_process):
    #                     num_RED_win, num_BLUE_win, trajectory = return_value[proc].get()
    #                     self.num_RED_win += num_RED_win
    #                     self.num_BLUE_win += num_BLUE_win
    #                     data = self.memory.load_process_data(proc)
    #                     re_data.extend(data)
    #                     re_trajectory.extend(trajectory)
    #             else:
    #                 # Single process
    #                 loss_player, data, trajectory, chess_record = self.game.play_one_game(in_layer=None, in_board=None,
    #                                                                                       in_player=None,
    #                                                                                       proc_id=self.args.SingleProcess,
    #                                                                                       num_iter=num_iter, mcts=self.mcts)
    #                 self.memory.save_chess_record(num_iter, chess_record)
    #                 re_data.extend(data)
    #                 re_trajectory.extend(trajectory)
    #
    #                 # 统计和棋数据在这里
    #                 if loss_player == self.game.BLUE:
    #                     self.num_RED_win += 1
    #                 elif loss_player == self.game.RED:
    #                     self.num_BLUE_win += 1
    #
    #         return re_data, re_trajectory
    #     # 从随机棋盘开始下
    #     else:
    #         boards = self.get_play_boards(num=self.args.num_play_game, num_iter=num_iter)
    #         Log.print_string('board length: ' + str(len(boards)))
    #         in_boards = boards[:self.args.num_play_game]
    #         for game_num, [layer, board, player] in enumerate(in_boards):
    #             Log.print_string('current data num: ' + str(len(re_data)) + ' !')
    #             if len(re_data) > self.args.min_data_num:
    #                 Log.print_string('The num of data reach the num: ' + str(self.args.min_data_num) + '; current game num: ' + str(game_num + 1) + '; Stop !')
    #                 return re_data, re_trajectory
    #             Log.print_play_num(game_num + 1)
    #             if self.args.multiprocess:
    #                 # Multi process
    #                 return_value = []
    #                 pool = multiprocessing.Pool(processes=self.args.num_process)
    #                 for proc in range(self.args.num_process):
    #                     return_value.append(pool.apply_async(self.open_multiprocess, (layer, board, player, proc, num_iter)))
    #                 pool.close()
    #                 pool.join()
    #
    #                 for proc in range(self.args.num_process):
    #                     num_RED_win, num_BLUE_win, trajectory = return_value[proc].get()
    #                     self.num_RED_win += num_RED_win
    #                     self.num_BLUE_win += num_BLUE_win
    #                     data = self.memory.load_process_data(proc)
    #                     re_data.extend(data)
    #                     re_trajectory.extend(trajectory)
    #             else:
    #                 # Single process
    #                 loss_player, data, trajectory, chess_record = self.game.play_one_game(in_layer=layer, in_board=board, in_player=player, proc_id=self.args.SingleProcess, num_iter=num_iter, mcts=self.mcts)
    #                 self.memory.save_chess_record(num_iter, chess_record)
    #                 re_data.extend(data)
    #                 re_trajectory.extend(trajectory)
    #                 # 统计和棋数据在这里
    #                 if loss_player == self.game.RED:
    #                     self.num_BLUE_win += 1
    #                 elif loss_player == self.game.BLUE:
    #                     self.num_RED_win += 1
    #         return re_data, re_trajectory

    def open_multiprocess(self, layers, board, player, proc: int, num_iter: int) -> [int, int, list]:
        trajectory = []
        if proc == self.args.print_log_process:
            sys.stdout = Logger(self.args.GAME_NAME, "log_process_" + str(self.args.print_log_process) + ".log", sys.stdout)
        num_BLUE_win, num_RED_win = 0, 0
        return_data = []
        loss_player, data, trajectory_data, chess_record = self.game.play_one_game(in_layer=layers, in_board=board, in_player=player, proc_id=proc, num_iter=num_iter, mcts=self.mcts)
        self.memory.save_chess_record(num_iter, chess_record)
        return_data.extend(data)
        trajectory.extend(trajectory_data)

        # 统计和棋数据在这里
        if loss_player == self.game.RED:
            self.num_BLUE_win += 1
        elif loss_player == self.game.BLUE:
            self.num_RED_win += 1

        self.memory.save_process_data(proc, return_data)
        return num_RED_win, num_BLUE_win, trajectory

    # def move_forward(self, num_iter, loss_p, loss_v):
    #     start = self.args.num_net * self.args.start_search_step_threshold
    #     end = self.args.num_net * (self.args.start_search_step_threshold + self.args.interval_search_step_threshold)
    #     interval = self.args.num_net * self.args.interval_search_step_threshold
    #     avg_loss = sum(loss_p[start: end]) / interval
    #     Log.print_string('avg_loss: ' + str(round(avg_loss, 4)))
    #     if (num_iter - self.args.old_num_iter) >= self.args.delta_num_iter and self.args.start_search_step_threshold > 0:
    #         self.args.old_num_iter = num_iter
    #         if self.args.start_search_step_threshold >= self.args.delta_search_step_threshold:
    #             self.args.start_search_step_threshold -= self.args.delta_search_step_threshold
    #         else:
    #             self.args.start_search_step_threshold = 0
    def move_forward(self, num_iter, loss_p, loss_v, root_Q_total_num, root_Q_right_num, search_step_root_Q_total_num,
                     search_step_root_Q_right_num):
        start = self.args.num_net * self.args.start_search_step_threshold
        end = self.args.num_net * self.args.num_max_layers
        interval = end - start
        avg_loss = sum(loss_p[start:]) / interval
        Log.print_string('avg_loss: ' + str(round(avg_loss, 4)))
        Q_rate = round(root_Q_right_num / root_Q_total_num, 4)
        search_step_Q_rate = round(search_step_root_Q_right_num / search_step_root_Q_total_num, 4)
        print("Q_rate: ", Q_rate)
        if (
                num_iter - self.args.old_num_iter) >= self.args.delta_num_iter and self.args.start_search_step_threshold > 0:
            if self.args.update_iter_by_Q and (
                    not Q_rate >= self.args.root_Q_threshold_rate or not search_step_Q_rate >= self.args.start_search_step_root_Q_threshold_rate):
                print("不满足Q约束，不前进！！！！！！")
                return None
            self.args.old_num_iter = num_iter
            if self.args.start_search_step_threshold >= self.args.delta_search_step_threshold:
                self.args.start_search_step_threshold -= self.args.delta_search_step_threshold
                print("满足Q约束，前进一步！！！！！！")
            else:
                self.args.start_search_step_threshold = 0