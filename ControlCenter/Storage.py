import os
import sys
from pickle import Pickler, Unpickler
import torch
import re
import numpy as np
import itertools
import time

BLACK = -2
WHITE = 2
EMPTY = 0
ARROW = 1


class Storage:
    """
    Store game tree, network model, game results and other information
    """

    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.trajectory = dict()

    def save_Q_threshold(self, num_iter, root_Q_total_num, root_Q_right_num, search_step_root_Q_total_num,
                         search_step_root_Q_right_num, start_search_step_threshold):
        """

        :return:
        """
        file_path = "./log/" + self.args.GAME_NAME + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        file_name = os.path.join(file_path, 'root_Q_record.csv')
        with open(file_name, "a") as file_point:
            if not file_point.tell():
                # file_point.write(f"Iter, Red, Blue, step_threshold, old_num_iter, old_avg_loss\n")
                file_point.write(
                    f"Iter, Root_Q_right_num, Root_Q_total_num, search_step_threshold, Q_Rate, Start_search_step_Q_rate\n")
            if root_Q_total_num == 0:
                root_Q_total_num = 1e-5
            if search_step_root_Q_total_num == 0:
                search_step_root_Q_total_num = 1e-5
            file_point.write(
                f"{num_iter}, {root_Q_right_num}, {root_Q_total_num}, {start_search_step_threshold}, {round(root_Q_right_num / root_Q_total_num, 4)}, {round(search_step_root_Q_right_num / search_step_root_Q_total_num, 4)}\n")

    def load_trajectory_replay_buffer(self):
        """
        Get data from storage and put it in the replay_buffer
        :return:
        """
        file_path = "./data/" + self.args.GAME_NAME + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        examples_file = os.path.join("./data/" + self.args.GAME_NAME + "/", 'trajectory.pth.tar.examples')
        if not os.path.isfile(examples_file):
            print("File with trajectory_replay_buffer not found. Continue? [y|n]")
            r = input()
            if r != "y":
                sys.exit()
        else:
            with open(examples_file, "rb") as f:
                self.trajectory = Unpickler(f).load()

    def save_trajectory_replay_buffer(self):
        """
        Store the data in the replay_buffer
        :return:
        """
        file_path = "./data/" + self.args.GAME_NAME + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = os.path.join(file_path, 'trajectory.pth.tar.examples')
        with open(file_name, "wb+") as f:
            Pickler(f).dump(self.trajectory)

    def load_boards_replay_buffer(self, replay_buffer: object) -> None:
        """
        Load the latest experience replay pool data
        :return:
        """
        examples_file = os.path.join("./data/" + self.args.GAME_NAME + "/", 'train_data.pth.tar.examples')
        if not os.path.isfile(examples_file):
            print("File with load_boards_replay_buffer not found. Continue? [y|n]")
            r = input()
            if r != "y":
                sys.exit()
        else:
            print("Find the train_data")
            with open(examples_file, "rb") as f:
                replay_buffer.total_data = Unpickler(f).load()

    def save_boards_replay_buffer(self, total_data):
        """
        Persistent storage of total training data
        :return:
        """
        file_path = "./data/" + self.args.GAME_NAME + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = os.path.join(file_path, 'train_data.pth.tar.examples')
        with open(file_name, "wb+") as f:
            Pickler(f).dump(total_data)

    def save_process_data(self, proc, return_data):
        file_path = "./data/" + self.args.GAME_NAME + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = os.path.join(file_path, 'process_' + str(proc) + '_data.pkl')
        with open(file_name, "wb+") as f:
            Pickler(f).dump(return_data)

    def load_process_data(self, proc):
        examples_file = os.path.join("./data/" + self.args.GAME_NAME + "/", 'process_' + str(proc) + '_data.pkl')
        if not os.path.isfile(examples_file):
            print("File with load_process_data not found. Continue? [y|n]")
            r = input()
            if r != "y":
                sys.exit()
        else:
            with open(examples_file, "rb") as f:
                data = Unpickler(f).load()
        return data

    def save_step_data_num(self, num_iter, step_data_num):
        """

        :return:
        """
        file_path = "./log/" + self.args.GAME_NAME + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = os.path.join(file_path, 'step_data_num.csv')
        with open(file_name, "a") as file_point:
            # file_point.write("iter: " + str(num_iter) + ' ' + str(list(step_data_num)) + '\n')
            file_point.write(f"{num_iter}, {str(list(step_data_num))[1:-1]}\n")

    def load_step_data_num(self):
        """

        :return:
        """
        file_path = "./log/" + self.args.GAME_NAME + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        with open("./log/" + self.args.GAME_NAME + "/" + "step_data_num.csv", "r") as file_point:
            lines = file_point.readlines()
            last_line = lines[-1]
            return list(np.where(np.array(re.findall(r"\d+\.?\d*", last_line)) == '0')[0])[-1]

    def load_vs_info(self):
        """

        :return:
        """
        file_path = "./log/" + self.args.GAME_NAME + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        with open("./log/" + self.args.GAME_NAME + "/" + "play_record.csv", "r") as file_point:
            lines = file_point.readlines()
            last_line = lines[-1]
            return [int(val) if i != 5 else float(val) for i, val in enumerate(re.findall(r"\d+\.?\d*", last_line))]

    def save_vs_info(self, num_iter, num_white_win, num_black_win, start_search_step_threshold, old_num_iter):
        """

        :return:
        """
        file_path = "./log/" + self.args.GAME_NAME + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        file_name = os.path.join(file_path, 'play_record.csv')
        with open(file_name, "a") as file_point:
            if not file_point.tell():
                # file_point.write(f"Iter, Red, Blue, step_threshold, old_num_iter, old_avg_loss\n")
                file_point.write(f"Iter, Red, Blue, search_step_threshold, old_num_iter, Red/(Blue + Red)\n")
            file_point.write(
                f"{num_iter}, {num_white_win}, {num_black_win}, {start_search_step_threshold}, {old_num_iter}, {round(num_white_win / (num_black_win + num_white_win), 4)}\n")

    def save_loss_step_p(self, loss):
        """
        Save the cross entropy of each step before each iteration of training
        """
        file_path = "./log/" + self.args.GAME_NAME + "/"
        file_name = os.path.join(file_path, 'loss_step_p.csv')
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            with open(file_name, "a") as file_point:
                file_point.write(str([i for i in range(1, len(loss) + 1)]) + '\n')
        with open(file_name, "a") as file_point:
            if not file_point.tell():
                pass
            file_point.write(str([round(i, 4) for i in loss])[1:-1] + '\n')

    def load_loss_step_p(self):
        file_path = "./log/" + self.args.GAME_NAME + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        with open("./log/" + self.args.GAME_NAME + "/" + "loss_step_p.csv", "r") as file_point:
            lines = file_point.readlines()
            last_line = lines[-1]
            return [float(i) for i in re.findall(r"\d+\.?\d*", last_line)]

    def save_loss_step_v(self, loss):
        """
        Save the cross entropy of each step before each iteration of training
        """
        file_path = "./log/" + self.args.GAME_NAME + "/"
        file_name = os.path.join(file_path, 'loss_step_v.csv')
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            with open(file_name, "a") as file_point:
                file_point.write(str([i for i in range(1, len(loss) + 1)]) + '\n')
        with open(file_name, "a") as file_point:
            file_point.write(str([round(i, 4) for i in loss])[1:-1] + '\n')

    def save_time_train(self, iter_num, time_game, time_train):
        """
        Save the time of training and generating data by playing self
        """
        file_path = "./log/" + self.args.GAME_NAME + "/"
        file_name = os.path.join(file_path, 'time_train.txt')
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            with open(file_name, "a") as file_point:
                file_point.write("Iter, Play, Train, Iter_total\n")

        with open(file_name, "a") as file_point:
            file_point.write("{iter_num}, {time_game}, {time_train}, {time_game+time_train}\n")
            file_point.write(
                "Iter " + str(iter_num) + " :" + "   Play:  " + str("%.2f" % time_game) + " s    Train:  " + str(
                    "%.2f" % time_train) + " s" +
                '   total:  ' + str("%.2f" % (time_game + time_train)) + " s" + '\n')

    def save_loss(self, pi_loss, v_loss, reg_loss):
        """
        保存loss值为txt文件格式
        :param pi_loss: float: 每次训练的 p + v 损失值
        :param v_loss: float: 每次训练的 p + v 损失值
        :param reg_loss: float: 每次训练的 p + v 损失值
        :return:
        """
        with open("./log/" + self.args.GAME_NAME + "/loss.txt", "a") as file_point:
            file_point.write(f"pi_loss:{pi_loss:6.3f}, v_loss:{v_loss:6.3f}, reg_loss:{reg_loss:6.4f}\n")

    def load_model_weight(self, num):
        """
        Load network parameters
        :param num:(str)
        :return:
        """
        file_path = "./data/" + self.args.GAME_NAME + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join("./data/" + self.args.GAME_NAME + "/", "model_" + num + ".pth.tar")
        if not os.path.exists(file_path):
            raise ("No model in path {}".format(file_path))
        map_location = None if self.args.cuda else 'cpu'
        model = torch.load(file_path, map_location=map_location)
        return model["state_dict"]

    def save_model_weight(self, net, num):
        """
        Store network parameters
        :param net:(object)
        :param num:(str)
        :return:
        """
        file_path = "./data/" + self.args.GAME_NAME + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join("./data/" + self.args.GAME_NAME + "/", "model_" + num + ".pth.tar")
        if not os.path.exists("./data/" + self.args.GAME_NAME + "/"):
            print("Checkpoint Directory does not exist! Making directory {}".format("./data/" + self.game.GAME_NAME))
            os.mkdir("./data/" + self.args.GAME_NAME + "/")
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            # 保存神经网络模型放到 checkpoint.pth.tar 目录中
            'state_dict': net.state_dict(),
        }, file_path)

    def save_chess_record(self, iter_num, game_num, record):
        record.append('')
        if self.args.GAME_NAME.split('_')[0] == 'amazons':
            file_path = "./log/" + self.args.GAME_NAME + "/play_record/"
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            file_name = os.path.join(file_path, 'record_' + str(iter_num) + '_' + str(game_num) + '.txt')
            with open(file_name, "a+", encoding="utf-8") as file_point:
                for i, item in enumerate(record):
                    if i == 0:
                        file_point.write(item)
                    else:
                        file_point.write('\n' + item)

    def load_total_train_data(self, replay_buffer):
        """

        :return:
        """
        examples_file = os.path.join("./data/" + self.args.GAME_NAME + "/", 'train_data.pth.tar.examples')
        if not os.path.isfile(examples_file):
            print("File with load_boards_replay_buffer not found. Continue? [y|n]")
            r = input()
            if r != "y":
                sys.exit()
        else:
            print("Find the train_data")
            with open(examples_file, "rb") as f:
                replay_buffer.total_data = Unpickler(f).load()

    def save_total_train_data(self, total_data):
        """

        :return:
        """
        file_path = "./data/" + self.args.GAME_NAME + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = os.path.join(file_path, 'train_data.pth.tar.examples')
        with open(file_name, "wb+") as f:
            Pickler(f).dump(total_data)

    def load_iter_train_data(self, replay_buffer):
        """

        :return:
        """
        examples_file = os.path.join("./data/" + self.args.GAME_NAME + "/", 'iter_train_data.pth.tar.examples')
        if not os.path.isfile(examples_file):
            print("File with iter_train_data not found. Continue? [y|n]")
            r = input()
            if r != "y":
                sys.exit()
        else:
            print("Find the train_data")
            with open(examples_file, "rb") as f:
                replay_buffer.iter_data = Unpickler(f).load()

    def save_iter_train_data(self, iter_data):
        """

        :return:
        """
        file_path = "./data/" + self.args.GAME_NAME + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = os.path.join(file_path, 'iter_train_data.pth.tar.examples')
        with open(file_name, "wb+") as f:
            Pickler(f).dump(iter_data)

    def save_train_time(self, iter_num, time_train):
        """
        Save the time of training and generating data by playing self
        """
        file_path = "./log/" + self.args.GAME_NAME + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = os.path.join(file_path, 'train_time.txt')
        with open(file_name, "a") as file_point:
            file_point.write("Iter " + str(iter_num) + " :" + "    Train:  " + str(
                "%.2f" % time_train) + " s" + '    Current Time: ' + time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                   time.localtime()) + '\n')

    def save_play_time(self, iter_num, time_play):
        """
        Save the time of training and generating data by playing self
        """
        file_path = "./log/" + self.args.GAME_NAME + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = os.path.join(file_path, 'play_time.txt')
        with open(file_name, "a") as file_point:
            file_point.write("Iter " + str(iter_num) + ":" + "    Play:  " + str(
                "%.2f" % time_play) + " s" + '    Current Time: ' + time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                  time.localtime()) + '\n')

    def load_model_data(self, file_name):
        folder = './data/' + self.args.GAME_NAME + "/"
        model_name = folder + file_name
        get_handle = open(model_name, 'rb')
        data = get_handle.read()
        get_handle.close()
        return data

    def save_p_v_dict_data(self, p_v_dict):
        """

        :return:
        """
        file_path = "./data/" + self.args.GAME_NAME + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = os.path.join(file_path, 'p_v_dict.pth.tar.examples')
        with open(file_name, "wb+") as f:
            Pickler(f).dump(p_v_dict)

    def load_p_v_dict_data(self, file_name):
        folder = "./data/" + self.args.GAME_NAME + "/"
        model_name = folder + file_name
        get_handle = open(model_name, 'rb')
        data = get_handle.read()
        get_handle.close()
        return data

class Log:
    """
    Print various log information
    """

    @staticmethod
    def print_string(strings):
        print(strings)

    @staticmethod
    def print_iter_num(num):
        """
        Print the total number of iterations of network training
        :param num:(int)
        :return:None
        """
        print()
        print('####################################  IterNum: ' + str(num) + ' ####################################')

    @staticmethod
    def print_play_num(num):
        """
        Print the number of games
        :param num:(int)
        :return:None
        """
        print()
        print('=====================================  Game: ' + str(num) + ' =====================================')

    @staticmethod
    def print_endgame():
        print()
        print('=====================================  Start End Game =====================================')

    @staticmethod
    def print_step_num(num):
        """
        Print the step of moves in a game_rules
        :param num:(int)
        :return:None
        """
        print('---------------------------  Step: ' + str(num) + ' ---------------------------')

    @staticmethod
    def print_vs_info(args, num_player_1, num_player_2):
        """
        Print winning or losing information
        :param args:(object)
        :param num_player_1:(int)
        :param num_player_2:(int)
        :return:None
        """
        player_1 = ""
        player_2 = ""
        if args.GAME_NAME.split('_')[0] == 'amazons':
            player_1 = "White "
            player_2 = "Black "
        elif args.GAME_NAME.split('_')[0] == 'go':
            player_1 = "White "
            player_2 = "Black "
        elif args.GAME_NAME.split('_')[0] == 'dotsboxes':
            player_1 = "RED "
            player_2 = "BLUE "
        print(player_1, 'win: ', num_player_1, '; ', player_2, 'win:', num_player_2)

    @staticmethod
    def print_game_result(player):
        """
        Print winning or losing information
        :return:None
        """
        player_str = ["BLACK", "WHITE"]
        print()
        print("===== ", player_str[int(0.5 + player / 4)], "lose =====")
        print("##### last board #####")

    @staticmethod
    def print_simulation_results(strings, simulation_n, simulation_q, simulation_u):
        print(strings)
        print('N: ', simulation_n)
        print('Q: ',simulation_q)
        print('U: ',simulation_u)

    def print_dots_and_boxes_board(self, board):
        board = board.copy()
        board = board.reshape(11, 11)
        dic = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "L"]
        print('\033[1;30;45m%s\033[0m' % '  0 1 2 3 4 5 6 7 8 9 0   ')
        print('\033[1;30;45m%s\033[0m' % '0 ', end="")
        # print(1, end="")
        for i in range(11):
            for j in range(11):
                # 领地
                if (i + 1) % 2 == 0 and (j + 1) % 2 == 0:
                    if board[i][j] == -2:
                        print('\033[1;34m%s\033[0m' % 'B', end=" ")
                    elif board[i][j] == 2:
                        print('\033[1;31m%s\033[0m' % 'R', end=" ")
                    else:
                        print('\033[1m%s\033[0m' % 'E', end=" ")
                # 点
                elif (i + 1) % 2 != 0 and (j + 1) % 2 != 0:
                    print('\033[1;33m%s\033[0m' % '⊙', end="")  # 显示方式：1； 　字体色：31；  背景色：46
                # 横边
                elif (i + 1) % 2 != 0 and (j + 1) % 2 == 0:
                    if board[i][j] == -2:
                        print('\033[1;34m%s\033[0m' % '———', end="")
                    elif board[i][j] == 2:
                        print('\033[1;31m%s\033[0m' % '———', end="")
                    else:
                        print("   ", end="")
                # 竖边
                elif (i + 1) % 2 == 0 and (j + 1) % 2 != 0:
                    if board[i][j] == -2:
                        print('\033[1;34m%s\033[0m' % '│', end=" ")
                    elif board[i][j] == 2:
                        print('\033[1;31m%s\033[0m' % '│', end=" ")
                    else:
                        print(" ", end=" ")
            if (i + 1) % 2 == 1:
                print(" ", end="")
            k = dic[i]
            k = k + " "
            # print('\033[1;30;45m%s\033[0m' % '  ', end="")
            print('\033[1;30;45m%s\033[0m' % k, end="")
            print("\n", end="")
            if i + 2 < 11:
                j = i + 1
                j = str(j) + " "
                print('\033[1;30;45m%s\033[0m' % j, end="")
            elif i + 2 == 10:
                print('\033[1;30;45m%s\033[0m' % '0 ', end="")
            elif i + 2 == 11:
                print('\033[1;30;45m%s\033[0m' % '0 ', end="")
        print('\033[1;30;45m%s\033[0m' % '  A B C D E F G H I J L   ')

    @staticmethod
    def print_board(board, board_size, next_action):
        """
        Print the board
        :param board: (np.array)
        :param board_size: (int)
        :param next_action: (tuple)

        :return:None
        """
        pretty_print_map = {
            WHITE: '\x1b[0;31;47mW',
            ARROW: '\x1b[0;31;46m#',
            EMPTY: '\x1b[0;31;43m.',
            BLACK: '\x1b[0;31;40mB',
        }
        board = np.copy(board)
        # 原始棋盘内容
        raw_board_contents = []
        for i in range(board_size):
            row = []
            for j in range(board_size):
                # 指明上一步走的位置：如果上一个playermove存在，且坐标和其对应的move相同，则添加< ； 否则添加空格
                appended = '<' if (i * board_size + j) in next_action else ' '
                row.append(pretty_print_map[board[i, j]] + appended)
                row.append('\x1b[0m')
            raw_board_contents.append(''.join(row))
        # 行标签 N~1
        row_labels = ['%2d ' % i for i in range(board_size, 0, -1)]
        # 带标注的每一行的内容
        annotated_board_contents = [''.join(r) for r in zip(row_labels, raw_board_contents, row_labels)]
        # 列标签
        header_footer_rows = ['   ' + ' '.join('ABCDEFGHIJ'[:board_size]) + '   ']
        # 带标注的棋盘
        # itertools.chain将不同容器中的元素连接起来，便于遍历
        annotated_board = '\n'.join(itertools.chain(header_footer_rows, annotated_board_contents, header_footer_rows))
        print(annotated_board)

    @staticmethod
    def print_args(args):
        """
        Print some parameter logs
        :param args:(object)
        :return:None
        """
        print("\n--------------------- Args ----------------------")
        print('| Iterate numbers: ', args.num_iter)
        print('| Iterate every ', args.num_play_game, ' games')
        print('| Real move policy : ', args.sharpening_policy_t)
        print('|................ Mcts params ................')
        print('| Train layers MCTS simulation times: ', args.train_num_search)
        print('| UCT function parameter Cpuct: ', args.Cpuct)
        print('|............. replay_buffer params .............')
        # print('| Threshold for N to extract data: ', args.N_threshold)
        # print('| Threshold for N+Q to extract data: ', args.N_Q_threshold1)
        print('| Threshold for replay buffer to delete:', args.replay_buffer_threshold)
        print('| replay_decay_rate: ', args.replay_decay_rate)
        print('|................ NetWork params ................')
        print('| load_latest_model: ', args.load_latest_model)
        print("| learning rate: ", args.lr)
        print("| The number of network parameters:", args.num_params)
        print('| Use GPU train: ', 'Yes' if args.cuda else 'No')
        print('|................ multiprocess params ................')
        if args.multiprocess:
            print('| Use Multi Process: ', 'Yes', ' ; Num Process: ', args.num_process)
        else:
            print('| Use Multi Process: ', 'No')
        print("-------------------------------------------------\n")

    @staticmethod
    def print_action_and_search_time(proc, player, actions, time_start, time_end):
        """
        Print MCTS search time

        :param proc:
        :param player:
        :param actions:
        :param time_start:
        :param time_end:
        :return:None
        """
        player_str = ["BLACK", "WHITE"]
        if proc == -1:
            print(player_str[int(0.5 + player / 4)], "moves: ", actions, ', search time: ', f"{time_end - time_start: .2f} s")

        else:
            print("Process " + str(proc) + ": " + player_str[int(0.5 + player / 4)], "moves: ", actions, ', search time: ', f"{time_end - time_start: .2f} s")

    @staticmethod
    def print_str_int(string, num):
        """
        Print strings and integers
        :param string:
        :param num:
        :return: None
        """
        print(string, ":", num)

    @staticmethod
    def print_action_and_search_time_dots(proc, player, actions, time_start, time_end):
        player_str = ["BLUE", "RED"]
        if proc == -1:
            print(player_str[int(0.5 + player / 4)], "moves: ", actions, ', search time: ', f"{time_end - time_start: .2f} s")

        else:
            print("Process " + str(proc) + ": " + player_str[int(0.5 + player / 4)], "moves: ", actions, ', search time: ', f"{time_end - time_start: .2f} s")

    @staticmethod
    def print_game_result_dots(player):
        player_str = ["BLUE", "RED"]
        print()
        print("===== ", player_str[int(0.5 + player / 4)], "lose =====")
        print("##### last board #####")


class Logger(object):
    """
    Persistent save terminal output to file
    """
    def __init__(self, game_name, filename, stream=sys.stdout):
        self.terminal = stream
        file_path = "./log/" + game_name + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        self.file_name = "./log/" + game_name + "/" + filename
        self.log = open(self.file_name, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log = open(self.file_name, 'a')
        self.log.write(message)
        self.log.close()

    def flush(self):
        pass


