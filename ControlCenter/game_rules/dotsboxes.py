import numpy as np
import time
import torch
from Storage import Log
from game_rules.abstract_game import AbstractGame

dx = [0, -1, 0, 1]
dy = [-1, 0, 1, 0]


# 点格棋棋规
class Game(AbstractGame):

    def __init__(self, args):
        super().__init__()
        self.args = args
        # 假设点格棋的双方分别是蓝色和红色
        self.EMPTY = 0
        self.BLUE = -2
        self.RED = 2
        self.BLUE_territory = -1
        self.RED_territory = 1
        # 红棋先手，蓝棋后手
        self.first_hand = self.RED
        self.second_hand = self.BLUE
        self.LOSE = -1
        self.WIN = 1
        self.GAME_NOT_END = 0
        self.GAME_END = -1
        self.side_size = args.side_size
        self.board_size = args.board_size
        self.action_size = self.board_size ** 2
        self.site_size = args.site_size
        self.Cpuct = args.Cpuct
        self.board2d_shape = (self.board_size, self.board_size)
        self.board1d_shape = (self.board_size ** 2,)
        self.player = None
        self.board = None
        # 所有的边
        self.all_sides_2D = []
        self.all_sides_1D = []
        # 所有的领地
        self.all_sites_2D = []
        self.all_sites_1D = []
        count = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                # 领地
                if (i + 1) % 2 == 0 and (j + 1) % 2 == 0:
                    self.all_sites_2D.append((i, j))
                    self.all_sites_1D.append(count)
                # 横边
                elif (i + 1) % 2 == 0 and (j + 1) % 2 != 0:
                    self.all_sides_2D.append((i, j))
                    self.all_sides_1D.append(count)
                # 竖边
                elif (i + 1) % 2 != 0 and (j + 1) % 2 == 0:
                    self.all_sides_2D.append((i, j))
                    self.all_sides_1D.append(count)
                count += 1
        self.init_board()

    # 初始化棋盘
    def init_board(self):
        # board_size = 11
        board_size = self.board_size
        # 点格棋的起手玩家是红方
        self.player = self.RED
        # 此处的board是11 * 11的二维矩阵，包括着点、边、和领地的三个信息在board里面
        # 新生成的board是11 * 11的原因是因为0行和0列是都不用的，只使用下标是0-10的部分
        self.board = np.zeros((board_size, board_size), dtype="int8")
        # 点格棋初始化不需要预先放置棋子，所以直接返回一维的棋盘数据
        return self.board.reshape((-1,))

    # 训练中进行一局博弈
    def play_one_game_new(self, in_layer, in_board, in_player, proc_id: int, num_iter: int,
                          mcts: object) -> [int, list, list, list]:
        self.args = mcts.args
        if in_player is not None:
            Log.print_string("layers: " + str(in_layer) + " player: " + ('RED' if in_player == self.RED else 'BLUE'))
        board = self.init_board() if in_board is None else np.array(in_board).reshape((-1,))
        player = self.player if in_player is None else in_player
        play_step = 0 if in_layer is None else in_layer // self.args.num_net
        return_data, trajectory = [], []
        # 初始化棋盘
        z = self.GAME_NOT_END
        record = ["#[AM][先手][后手][后手胜][2050/12/31 23:59:59 河北][国赛];"]
        use_prune = False
        if self.args.start_search_step_threshold > self.args.delta_search_step_threshold:
            use_prune = True
        last_move = None
        root_Q_total_num = 0  # 根节点总数
        root_Q_right_num = 0  # 根节点满足Q值传递约束的总数
        search_step_root_Q_total_num, search_step_root_Q_right_num = 0, 0
        while z == self.GAME_NOT_END:
            start_time = time.time()
            if proc_id == self.args.SingleProcess or proc_id == self.args.print_log_process:
                # 打印step数
                Log.print_step_num(play_step)
                # 打印棋盘
                self.print_dots_and_boxes_board(board.reshape(self.board_size, self.board_size))
                # 转换视角（统一使用某一种颜色进行神经网络的训练，比如红色，所以这里需要转换视角）
                # 蓝色方的话 需要转换成红色的视角进行落子操作
                transformed_board = self.change_perspectives(board, player)
                is_search = False
                if play_step < self.args.start_search_step_threshold:  # 用先验知识走
                    self.args.train_num_search, self.args.train_num_search_before = self.args.train_num_search_before, self.args.train_num_search
                    self.args.N_threshold, self.args.N_threshold_before = self.args.N_threshold_before, self.args.N_threshold

                    next_action = self.get_legal_action_dots_by_prune(transformed_board)
                    next_action = np.random.choice(next_action)

                    action, _, _, root_can_quick_move = mcts.select_action_dots(-proc_id,
                                                                                transformed_board,
                                                                                num_iter,
                                                                                play_step,
                                                                                "sharpening policy",
                                                                                False, last_move)
                    self.args.train_num_search, self.args.train_num_search_before = self.args.train_num_search_before, self.args.train_num_search
                    self.args.N_threshold, self.args.N_threshold_before = self.args.N_threshold_before, self.args.N_threshold
                    if root_can_quick_move:
                        next_action = action
                    else:
                        print("net_work choose action: ", (next_action//11, next_action % 11))
                else:  # 真实走
                    next_action, data, trajectory_data, root_can_quick_move = mcts.select_action_dots(proc_id,
                                                                                                      transformed_board,
                                                                                                      num_iter,
                                                                                                      play_step,
                                                                                                      "sharpening policy",
                                                                                                      False, last_move)
                    is_search = True
                    # return_data 增加数据
                    if data is not None:
                        return_data.extend(data)
                    else:
                        print("data is None")
                    # trajectory 增加数据
                    if trajectory_data is not None:
                        trajectory.extend(trajectory_data)
                    else:
                        print("trajectory_data is None")
                if play_step != 0:
                    print(mcts.root.dfs_move)
                if root_can_quick_move:  # 快速走几步
                    print("=" * 50)
                    print("此处因为open,快速走的Action~~~~~~~~~~~~~~~: ")
                    print([(i // self.board_size, i % self.board_size) for i in next_action])
                    print("=" * 50)
                    for i in next_action:
                        board = self.update_territory(board, (i // self.board_size, i % self.board_size), player)
                    last_move = next_action[-1]
                    z = self.game_over_dots(board)
                    play_step += len(next_action)
                    player = -player
                    number_of_territory_after, number_of_territory_before = 1, 0
                    if mcts.root.fake_last_move is not None:
                        last_move = mcts.root.fake_last_move
                else:
                    if is_search:
                        _, v = mcts.trainer.predict(transformed_board, mcts.trainer.net_work)
                        if self.args.start_search_step_threshold + self.args.start_search_step_root_Q_threshold_interval <= play_step:
                            root_Q_total_num += 1
                            search_step_root_Q_total_num += 1
                            if abs(v) >= self.args.root_Q_threshold and mcts.root.my_q * v > 0:
                                root_Q_right_num += 1
                            if abs(mcts.root.my_q) >= self.args.start_search_step_root_Q_threshold:
                                search_step_root_Q_right_num += 1
                    # 获取时间（结束时间）
                    last_move = next_action
                    end_time = time.time()
                    if proc_id == self.args.SingleProcess or proc_id == self.args.print_log_process:
                        # 打印结果和搜索的时间(点格棋自己写)
                        Log.print_action_and_search_time_dots(proc_id, player, next_action, start_time, end_time)
                        print("=" * 50)
                        print("Action 的 二维坐标的位置是：", self.all_sides_2D[self.all_sides_1D.index(next_action)])
                        print("=" * 50)
                    # 落边之前当前玩家的领地数量
                    number_of_territory_before = self.get_number_of_territory(board, player)
                    # 落边，进入下一个棋盘状态（player已经换成对方）
                    board, player = self.get_next_state(board, player, next_action)
                    # 因为此时玩家已经转换，所以需要加个负号 进行转换
                    number_of_territory_after = self.get_number_of_territory(board, -player)
                    # 判断对弈是否结束
                    z = self.game_over_dots(board)
                    play_step += 1
                # 出现了领地的增强 而且 对弈尚未结束
                if number_of_territory_after > number_of_territory_before and z == self.GAME_NOT_END:
                    # 将玩家转换为上一手玩家
                    player = -player
                    # 标记变量
                    flag = 1
                    # 再进行循环判断停止与否
                    while flag == 1 and z == self.GAME_NOT_END:
                        print("*" * 22 + "此处是因为连成，而走的一条线" + "*" * 22)
                        Log.print_step_num(play_step)
                        print("*" * 22 + "此处是因为连成，而走的一条线" + "*" * 22)
                        # 打印棋盘
                        self.print_dots_and_boxes_board(board.reshape(self.board_size, self.board_size))
                        # 依然要将棋盘转换视角
                        transformed_board = self.change_perspectives(board, player)

                        # 当前玩家落边之前的领地数量
                        number_of_territory_before = self.get_number_of_territory(board, player)
                        start_time = time.time()
                        is_search = False
                        if play_step < self.args.start_search_step_threshold:  # 用先验知识走
                            self.args.train_num_search, self.args.train_num_search_before = self.args.train_num_search_before, self.args.train_num_search
                            self.args.N_threshold, self.args.N_threshold_before = self.args.N_threshold_before, self.args.N_threshold

                            next_action = self.get_legal_action_dots_by_prune(transformed_board)
                            next_action = np.random.choice(next_action)
                            action, _, _, root_can_quick_move = mcts.select_action_dots(-proc_id,
                                                                                        transformed_board,
                                                                                        num_iter,
                                                                                        play_step,
                                                                                        "sharpening policy",
                                                                                        False, last_move)
                            self.args.train_num_search, self.args.train_num_search_before = self.args.train_num_search_before, self.args.train_num_search
                            self.args.N_threshold, self.args.N_threshold_before = self.args.N_threshold_before, self.args.N_threshold
                            if root_can_quick_move:
                                next_action = action
                            else:
                                print("net_work choose action: ", (next_action // 11, next_action % 11))
                        else:  # mct500搜索
                            next_action, data, trajectory_data, root_can_quick_move = mcts.select_action_dots(proc_id,
                                                                                                              transformed_board,
                                                                                                              num_iter,
                                                                                                              play_step,
                                                                                                              "sharpening policy",
                                                                                                              False,
                                                                                                              last_move)
                            is_search = True
                            # return_data 增加数据
                            if data is not None:
                                return_data.extend(data)
                            else:
                                print("data is None")
                            # trajectory 增加数据
                            if trajectory_data is not None:
                                trajectory.extend(trajectory_data)
                            else:
                                print("trajectory_data is None")
                        if play_step != 0:
                            print(mcts.root.dfs_move)
                        if root_can_quick_move:  # 快速走几步
                            print("=" * 50)
                            print("此处因为open,快速走的Action~~~~~~~~~~~~~~~: ")
                            print([(i // self.board_size, i % self.board_size) for i in next_action])
                            print("=" * 50)
                            for i in next_action:
                                board = self.update_territory(board, (i // self.board_size, i % self.board_size),
                                                              player)
                            last_move = next_action[-1]
                            z = self.game_over_dots(board)
                            play_step += len(next_action)
                            if mcts.root.fake_last_move is not None:
                                last_move = mcts.root.fake_last_move
                        else:
                            if is_search:
                                # root_Q_total_num += 1
                                _, v = mcts.trainer.predict(transformed_board, mcts.trainer.net_work)
                                if self.args.start_search_step_threshold + self.args.start_search_step_root_Q_threshold_interval <= play_step:
                                    root_Q_total_num += 1
                                    search_step_root_Q_total_num += 1
                                    if abs(v) >= self.args.root_Q_threshold and mcts.root.my_q * v > 0:
                                        root_Q_right_num += 1
                                    if abs(mcts.root.my_q) >= self.args.start_search_step_root_Q_threshold:
                                        search_step_root_Q_right_num += 1
                            last_move = next_action
                            end_time = time.time()
                            # 打印结果和搜索的时间(点格棋自己写)
                            Log.print_action_and_search_time_dots(proc_id, player, next_action, start_time, end_time)
                            print("=" * 50)
                            print("Action 的 二维坐标的位置是：",
                                  self.all_sides_2D[self.all_sides_1D.index(next_action)])
                            print("=" * 50)
                            # 落边，进入下一个棋盘状态
                            board, player = self.get_next_state(board, player, next_action)
                            # 因为此时玩家已经转换，所以需要加个负号（转换成上一手玩家）进行判断领地数量
                            number_of_territory_after = self.get_number_of_territory(board, -player)
                            # 没有发生领地的增加
                            if number_of_territory_after == number_of_territory_before:
                                flag = 0
                            else:
                                # 又产生了新的领地，玩家依旧得换回来
                                player = -player
                                flag = 1
                            # 还是需要判断下对弈是否结束
                            z = self.game_over_dots(board)
                            # step + 1
                            play_step += 1
                # GAME_END 的 值为 -1，z 是 game_over判断方法的返回值，若z = -1 / 1，则一轮博弈结束
                if z != self.GAME_NOT_END:
                    # SingleProcess = -1 && print_log_process = 0
                    # 获取到winner
                    winner = self.get_winner(board)
                    if proc_id == self.args.SingleProcess or proc_id == self.args.print_log_process:
                        print("root_Q_total_num: ", root_Q_total_num)
                        print("root_Q_right_num: ", root_Q_right_num)
                        print("start_search_step_root_Q_total_num: ", search_step_root_Q_total_num)
                        print("start_search_step_root_Q_right_num: ", search_step_root_Q_right_num)
                        # 因为print_game_result_dots 是 打印的谁输了，所以这里winner前面要加个负号
                        Log.print_game_result_dots(-winner)
                        # Log.print_board(board.reshape(self.board_size, self.board_size), self.board_size, next_action)
                        self.print_dots_and_boxes_board(board)
                    # TEST
                    # assert 0
                    # 统计统计的是谁败了，所以winner前面加个负号
                    return -winner, return_data, trajectory, record, root_Q_total_num, root_Q_right_num, search_step_root_Q_total_num, search_step_root_Q_right_num

    def play_one_game(self, in_layer: int, in_board: list, in_player: int, proc_id: int, num_iter: int,
                      mcts: object) -> [int, list, list, list]:
        if in_player is not None:
            Log.print_string("layers: " + str(in_layer) + " player: " + ('RED' if in_player == self.RED else 'BLUE'))
        board = self.init_board() if in_board is None else np.array(in_board).reshape((-1,))
        player = self.player if in_player is None else in_player
        play_step = 0 if in_layer is None else in_layer // self.args.num_net
        return_data, trajectory = [], []
        # 初始化棋盘
        z = self.GAME_NOT_END
        record = ["#[AM][先手][后手][后手胜][2050/12/31 23:59:59 河北][国赛];"]
        last_move = None  # 上一条边，mct要
        while z == self.GAME_NOT_END:
            start_time = time.time()
            if proc_id == self.args.SingleProcess or proc_id == self.args.print_log_process:
                # 打印step数
                Log.print_step_num(play_step)
                # 打印棋盘
                self.print_dots_and_boxes_board(board.reshape(self.board_size, self.board_size))
                # 转换视角（统一使用某一种颜色进行神经网络的训练，比如红色，所以这里需要转换视角）
                # 蓝色方的话 需要转换成红色的视角进行落子操作
                transformed_board = self.change_perspectives(board, player)

                # 获取到蒙特卡洛搜索树的仿真结果[送进去的棋盘永远是以红色方视角走边的]
                next_action, data, trajectory_data, root_can_quick_move = mcts.select_action_dots(proc_id,
                                                                                                  transformed_board,
                                                                                                  num_iter,
                                                                                                  play_step,
                                                                                                  "sharpening policy",
                                                                                                  last_move=last_move)
                if root_can_quick_move:  # 快速走几步
                    print("=" * 50)
                    print("此处因为open,快速走的Action~~~~~~~~~~~~~~~: ")
                    print([(i // self.board_size, i % self.board_size) for i in next_action])
                    print("=" * 50)
                    for i in next_action:
                        board = self.update_territory(board, (i // self.board_size, i % self.board_size), player)
                    last_move = next_action[-1]
                    z = self.game_over_dots(board)
                    play_step += len(next_action)
                    player = -player
                    number_of_territory_after, number_of_territory_before = 1, 0
                else:
                    last_move = next_action
                    # return_data 增加数据
                    return_data.extend(data)
                    # trajectory 增加数据
                    trajectory.extend(trajectory_data)
                    # 获取时间（结束时间）
                    end_time = time.time()
                    if proc_id == self.args.SingleProcess or proc_id == self.args.print_log_process:
                        # 打印结果和搜索的时间(点格棋自己写)
                        Log.print_action_and_search_time_dots(proc_id, player, next_action, start_time, end_time)
                        print("=" * 50)
                        print("Action 的 二维坐标的位置是：", self.all_sides_2D[self.all_sides_1D.index(next_action)])
                        print("=" * 50)
                    # 落边之前当前玩家的领地数量
                    number_of_territory_before = self.get_number_of_territory(board, player)
                    # 落边，进入下一个棋盘状态（player已经换成对方）
                    board, player = self.get_next_state(board, player, next_action)
                    # 因为此时玩家已经转换，所以需要加个负号 进行转换
                    number_of_territory_after = self.get_number_of_territory(board, -player)
                    # 判断对弈是否结束
                    z = self.game_over_dots(board)
                    play_step += 1
                # 出现了领地的增强 而且 对弈尚未结束
                if number_of_territory_after > number_of_territory_before and z == self.GAME_NOT_END:
                    # 将玩家转换为上一手玩家
                    player = -player
                    # 标记变量
                    flag = 1
                    # 再进行循环判断停止与否
                    while flag == 1 and z == self.GAME_NOT_END:
                        print("*" * 22 + "此处是因为连成，而走的一条线" + "*" * 22)
                        Log.print_step_num(play_step)
                        print("*" * 22 + "此处是因为连成，而走的一条线" + "*" * 22)
                        # 打印棋盘
                        self.print_dots_and_boxes_board(board.reshape(self.board_size, self.board_size))
                        # 依然要将棋盘转换视角
                        transformed_board = self.change_perspectives(board, player)

                        # 当前玩家落边之前的领地数量
                        number_of_territory_before = self.get_number_of_territory(board, player)
                        start_time = time.time()
                        # 此处传入的board应该是落完边的board
                        next_action, data, trajectory_data, root_can_quick_move = mcts.select_action_dots(proc_id,
                                                                                                          transformed_board,
                                                                                                          num_iter,
                                                                                                          play_step,
                                                                                                          "sharpening policy",
                                                                                                          last_move=last_move)
                        if root_can_quick_move:  # 快速走几步
                            print("=" * 50)
                            print("此处因为open,快速走的Action~~~~~~~~~~~~~~~: ")
                            print([(i // self.board_size, i % self.board_size) for i in next_action])
                            print("=" * 50)
                            for i in next_action:
                                board = self.update_territory(board, (i // self.board_size, i % self.board_size),
                                                              player)
                            last_move = next_action[-1]
                            z = self.game_over_dots(board)
                            play_step += len(next_action)
                        else:
                            last_move = next_action
                            # return_data 增加数据
                            return_data.extend(data)
                            # trajectory 增加数据
                            trajectory.extend(trajectory_data)
                            end_time = time.time()
                            # 打印结果和搜索的时间(点格棋自己写)
                            Log.print_action_and_search_time_dots(proc_id, player, next_action, start_time, end_time)
                            print("=" * 50)
                            print("Action 的 二维坐标的位置是：",
                                  self.all_sides_2D[self.all_sides_1D.index(next_action)])
                            print("=" * 50)
                            # 落边，进入下一个棋盘状态
                            board, player = self.get_next_state(board, player, next_action)
                            # 因为此时玩家已经转换，所以需要加个负号（转换成上一手玩家）进行判断领地数量
                            number_of_territory_after = self.get_number_of_territory(board, -player)
                            # 没有发生领地的增加
                            if number_of_territory_after == number_of_territory_before:
                                flag = 0
                            else:
                                # 又产生了新的领地，玩家依旧得换回来
                                player = -player
                                flag = 1
                            # 还是需要判断下对弈是否结束
                            z = self.game_over_dots(board)
                            # step + 1
                            play_step += 1
                # GAME_END 的 值为 -1，z 是 game_over判断方法的返回值，若z = -1 / 1，则一轮博弈结束
                if z != self.GAME_NOT_END:
                    # SingleProcess = -1 && print_log_process = 0
                    # 获取到winner
                    winner = self.get_winner(board)
                    if proc_id == self.args.SingleProcess or proc_id == self.args.print_log_process:
                        # 因为print_game_result_dots 是 打印的谁输了，所以这里winner前面要加个负号
                        Log.print_game_result_dots(-winner)
                        # Log.print_board(board.reshape(self.board_size, self.board_size), self.board_size, next_action)
                        self.print_dots_and_boxes_board(board)
                    # 统计统计的是谁败了，所以winner前面加个负号
                    return -winner, return_data, trajectory, record

    # 针对生成残局写的函数，主要是为了保留玩家信息在return_data里面
    def play_one_game_to_get_boards(self, in_layer: int, in_board: list, in_player: int, proc_id: int, num_iter: int,
                                    mcts: object) -> [int, list, list, list]:
        if in_player is not None:
            Log.print_string("layers: " + str(in_layer) + " player: " + ('RED' if in_player == self.RED else 'BLUE'))
        board = self.init_board() if in_board is None else np.array(in_board).reshape((-1,))
        player = self.player if in_player is None else in_player
        play_step = 0 if in_layer is None else in_layer // self.args.num_net
        return_data = []
        # 初始化棋盘
        z = self.GAME_NOT_END
        while z == self.GAME_NOT_END:
            start_time = time.time()
            if proc_id == self.args.SingleProcess or proc_id == self.args.print_log_process:
                Log.print_step_num(play_step)
                self.print_dots_and_boxes_board(board.reshape(self.board_size, self.board_size))
                # 转换视角（统一使用某一种颜色进行神经网络的训练，比如红色，所以这里需要转换视角）
                transformed_board = self.change_perspectives(board, player)
                # 获取到蒙特卡洛搜索树的仿真结果
                # next_action = ((pointx, pointy), (sidex, sidey))
                next_action, data, _ = mcts.select_action_dots(proc_id, transformed_board, num_iter, play_step,
                                                               "sharpening policy")
                # 先删除不是这个layers对应的（其他layers的玩家也没办法保存）
                new_data = []
                for i in range(len(data)):
                    num_iter, layers, my_n, board_data = data[i]
                    if layers == play_step:
                        data[i].append(player)
                        new_data.append(data[i])
                data = new_data.copy()
                # return_data 增加数据
                return_data.extend(data)
                # # return_data 增加玩家(这个时候的玩家还没有进行转换)
                # return_data.extend([player])
                # 获取时间（结束时间）
                end_time = time.time()
                if proc_id == self.args.SingleProcess or proc_id == self.args.print_log_process:
                    # 打印结果和搜索的时间(点格棋自己写)
                    Log.print_action_and_search_time_dots(proc_id, player, next_action, start_time, end_time)
                    print("=" * 50)
                    print("Action 的 二维坐标的位置是：", self.all_sides_2D[self.all_sides_1D.index(next_action)])
                    print("=" * 50)
                number_of_territory_before = self.get_number_of_territory(board, player)
                # 落子，进入下一个棋盘状态(之后的player就是需要落子的玩家)
                board, player = self.get_next_state(board, player, next_action)
                # 因为此时玩家已经转换，所以需要加个负号（转换成上一手玩家）进行判断
                number_of_territory_after = self.get_number_of_territory(board, -player)
                # 判断对弈是否结束
                z = self.game_over_dots(board)
                play_step += 1
                # 出现了领地的增强而且对弈尚未结束
                if number_of_territory_after > number_of_territory_before and z == self.GAME_NOT_END:
                    # 将玩家转换为上一手玩家
                    player = -player
                    # 标记变量
                    flag = 1
                    # 再进行循环判断停止与否
                    while flag == 1 and z == self.GAME_NOT_END:
                        print("*" * 22 + "此处是因为连成，而走的一条线" + "*" * 22)
                        Log.print_step_num(play_step)
                        print("*" * 22 + "此处是因为连成，而走的一条线" + "*" * 22)
                        # print("上一手玩已经再走了一步！棋盘打印如下")
                        self.print_dots_and_boxes_board(board.reshape(self.board_size, self.board_size))
                        # 依然要将棋盘转换视角
                        transformed_board = self.change_perspectives(board, player)
                        number_of_territory_before = self.get_number_of_territory(board, player)
                        start_time = time.time()
                        # 此处传入的board应该是落完子的board
                        next_action, data, trajectory_data = mcts.select_action_dots(proc_id, transformed_board,
                                                                                     num_iter, play_step,
                                                                                     "sharpening policy")
                        # 先删除不是这个layers对应的（其他layers的玩家也没办法保存）
                        new_data = []
                        for i in range(len(data)):
                            num_iter, layers, my_n, board_data = data[i]
                            if layers == play_step:
                                data[i].append(player)
                                new_data.append(data[i])
                        data = new_data.copy()
                        # return_data 增加数据
                        return_data.extend(data)
                        # # return_data 增加玩家(这个时候的玩家还没有进行转换)
                        # return_data.extend([player])
                        end_time = time.time()
                        # 打印结果和搜索的时间(点格棋自己写)
                        Log.print_action_and_search_time_dots(proc_id, player, next_action, start_time, end_time)
                        print("=" * 50)
                        print("Action 的 二维坐标的位置是：", self.all_sides_2D[self.all_sides_1D.index(next_action)])
                        print("=" * 50)
                        # 落子，进入下一个棋盘状态(之后的player就是需要落子的玩家)
                        board, player = self.get_next_state(board, player, next_action)
                        # 因为此时玩家已经转换，所以需要加个负号（转换成上一手玩家）进行判断
                        number_of_territory_after = self.get_number_of_territory(board, -player)
                        # 没有发生领地的增加
                        if number_of_territory_after == number_of_territory_before:
                            flag = 0
                        else:
                            # 产生了新的领地
                            # 玩家依旧得换回来
                            player = -player
                            flag = 1
                        z = self.game_over_dots(board)
                        play_step += 1
                # GAME_END的值为-1，z是game_over判断方法的返回值，若z=-1，则一轮博弈结束
                if z != self.GAME_NOT_END:
                    # SingleProcess = -1 && print_log_process = 0
                    # 获取到winner
                    winner = self.get_winner(board)
                    if proc_id == self.args.SingleProcess or proc_id == self.args.print_log_process:
                        # 因为print_game_result_dots是打印的谁输了，所以这里winner前面要加个负号
                        Log.print_game_result_dots(-winner)
                        # Log.print_board(board.reshape(self.board_size, self.board_size), self.board_size, next_action)
                        self.print_dots_and_boxes_board(board)
                    return return_data

    # 根据player 和 action 将 board 进入到下一个状态
    def get_next_state(self, board, player, action):
        # side都是一维的格式，可以转成二维的格式，然后进行对棋盘的修改操作
        # 两个assert debug
        assert type(board) is np.ndarray
        assert player in (self.BLUE, self.RED)
        # 不直接修改原棋盘 浅拷贝一个新的棋盘 然后进行操作
        board_temp = board.copy()
        board_temp = board_temp.reshape(self.board_size, self.board_size)
        # 将一维的边转换成二维的边
        side = self.all_sides_2D[self.all_sides_1D.index(action)]
        # 将棋盘的对应位置置为该玩家
        board_temp[side[0]][side[1]] = player
        # 每次有玩家落完边，更新领域
        board_temp = self.update_territory(board_temp, (side[0], side[1]), player)
        # 返回操作之后的棋盘 以及 对手玩家
        return board_temp, -player

    def judge_side_type(self, x, y, board):
        assert (x + y) % 2 == 1
        # 竖边
        sx = [1, 0, -1, -1, 0, 1]
        sy = [-1, -2, -1, 1, 2, 1]
        # 横边
        hx = [-1, -2, -1, 1, 2, 1]
        hy = [-1, 0, 1, 1, 0, -1]
        side_cnt = 0
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
                return 1, second_cnt
            elif x == self.board_size - 1:  # 下边界
                for k in range(3):
                    nx = x + hx[k]
                    ny = y + hy[k]
                    if board_temp[nx][ny] != 0:
                        first_cnt += 1
                return first_cnt, 1
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
                return 1, second_cnt
            elif y == self.board_size - 1:
                for k in range(3):
                    nx = x + sx[k]
                    ny = y + sy[k]
                    if board_temp[nx][ny] != 0:
                        first_cnt += 1
                return first_cnt, 1
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

    # 贪心剪枝，用于前部分迭代，生成高质量棋盘
    def get_legal_action_dots_by_prune(self, board):
        board_temp = board.copy()
        board_temp = board_temp.reshape((-1))

        side_3, side_12, side_4, side_44 = [], [], [], []
        occupied_side = 0
        available_sides = []
        # # 遍历棋盘先获取到所有的可行边
        for i in range(len(self.all_sides_1D)):
            if board_temp[self.all_sides_1D[i]] == self.EMPTY:
                available_sides.append(self.all_sides_1D[i])
                x = self.all_sides_1D[i] // self.board_size
                y = self.all_sides_1D[i] % self.board_size
                side = self.judge_side_type(x, y, board_temp)  # Example: side = (1,2)
                assert 1 <= side[0] <= 4 and 1 <= side[1] <= 4
                if side == (4, 4):
                    return [self.all_sides_1D[i]]
                if 4 in side:
                    side_4.append(self.all_sides_1D[i])
                elif 3 in side:
                    side_3.append(self.all_sides_1D[i])
                else:
                    side_12.append(self.all_sides_1D[i])
            else:
                occupied_side += 1
                # 有3个4的情况必在4中选
        if len(side_4) > 2:
            return side_4
        elif len(side_12) > 0:
            return side_12
        else:
            side_3.extend(side_4)
            return side_3
        # return available_sides

    # 获取棋盘上的所有合理走边位置
    def get_legal_action_dots(self, board):
        # 为避免修改原始棋盘 所以依然是浅拷贝一个原始棋盘
        board_temp = board.copy()
        board_temp = board_temp.reshape((-1))
        # 所有的可行边
        available_sides = []
        # side_4 = []
        # 遍历棋盘先获取到所有的可行边
        for i in range(len(self.all_sides_1D)):
            if board_temp[self.all_sides_1D[i]] == self.EMPTY:
                available_sides.append(self.all_sides_1D[i])
        return available_sides

    # 判断对弈是否结束【训练时使用】
    def game_over_dots(self, board):  # 训练使默认当前红色方
        board_temp = board.copy()
        board_temp = board_temp.reshape(self.board_size, self.board_size)
        territory_of_blue = 0
        territory_of_red = 0
        side_num = 0  # 占领的边的数量
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i + j) % 2 == 1 and board_temp[i][j] != 0:
                    side_num += 1
                if (i + 1) % 2 == 0 and (j + 1) % 2 == 0:
                    if board_temp[i][j] == self.RED_territory:
                        territory_of_red += 1
                    if board_temp[i][j] == self.BLUE_territory:
                        territory_of_blue += 1
                    if territory_of_red > (self.site_size // 2):
                        return self.WIN
                    if territory_of_blue > (self.site_size // 2):
                        return self.LOSE

        # 上面没有返回，证明还没结束
        return self.GAME_NOT_END

    def pd_unopened_game(self, board):  # edg_cnt只能为2和4
        board_temp = np.array(board).reshape(self.args.board_size, self.args.board_size)
        for i in range(self.args.board_size):
            for j in range(self.args.board_size):
                if i % 2 == 0 or j % 2 == 0:
                    continue
                edg_cnt = 0
                for k in range(4):
                    nx = i + dx[k]
                    ny = j + dy[k]
                    if board_temp[nx][ny] != 0:
                        edg_cnt += 1
                if edg_cnt == 0 or edg_cnt == 3 or edg_cnt == 1:
                    return False
        return True

    # 转换视角
    def change_perspectives(self, board, player):
        # 红棋不用训练转换视角，因为神经网络打算采用红棋视角训练
        if player == self.RED:
            return board
        # 否则需要转换视角
        board_temp = board.copy()
        board_temp = -board_temp
        return board_temp

    # to——string方法
    def to_string(self, board):
        board = np.array(board).reshape(self.board_size, self.board_size)
        red_territory, blue_territory = 0, 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == 1:
                    red_territory += 1
                elif board[i][j] == -1:
                    blue_territory += 1
                board[i][j] = abs(board[i][j])
        board = board.reshape(-1)
        board = np.append(board, red_territory - blue_territory)
        return board.tostring()

    # 数据增强 以及 生成训练数据
    def board_flip(self, board, pi, pos=None):
        board.shape = self.board2d_shape
        board_list = []
        new_pos = None
        # 1
        new_b = np.reshape(board, self.board2d_shape)
        pi_ = np.reshape(pi, self.board2d_shape)
        if pos is None:
            board_list += [(new_b, list(pi_.ravel()))]
        else:
            new_pos = np.zeros_like(pi_)
            new_pos[pos // self.board_size][pos % self.board_size] = 1
            board_list += [(new_b, new_pos, list(pi_.ravel()))]
        # 2
        new_b = np.fliplr(new_b)
        new_pi = np.fliplr(pi_)
        if pos is None:
            board_list += [(new_b, list(new_pi.ravel()))]
        else:
            new_pos = np.fliplr(new_pos)
            board_list += [(new_b, new_pos, list(new_pi.ravel()))]
        # 3
        new_b = np.flipud(new_b)
        new_pi = np.flipud(new_pi)
        if pos is None:
            board_list += [(new_b, list(new_pi.ravel()))]
        else:
            new_pos = np.flipud(new_pos)
            board_list += [(new_b, new_pos, list(new_pi.ravel()))]
        # 4
        new_b = np.fliplr(new_b)
        new_pi = np.fliplr(new_pi)
        if pos is None:
            board_list += [(new_b, list(new_pi.ravel()))]
        else:
            new_pos = np.fliplr(new_pos)
            board_list += [(new_b, new_pos, list(new_pi.ravel()))]

        board.shape = self.board1d_shape
        return board_list

    # 进入下一个状态，mcts部分的虚拟棋盘
    def get_next_board(self, board, side_choose=None):
        assert type(board) is np.ndarray
        # 浅拷贝一个棋盘
        board_temp = board.copy()
        board_temp = board_temp.reshape(self.board_size, self.board_size)
        # 这个地方考虑点格棋的特殊情况（点格棋的坐标是二维的）
        assert board_temp.shape == self.board2d_shape
        # side的情况， side_choose是一维的形式，需要转换成二维的
        side_choose = self.all_sides_2D[self.all_sides_1D.index(side_choose)]
        sidex = side_choose[0]
        sidey = side_choose[1]
        assert (sidex, sidey) in self.all_sides_2D
        # 落边之前的领地数量
        number_of_territory_before = self.get_number_of_territory(board_temp, self.RED)
        # 因为神经网络是使用红色方训练的，所以 每次落子 其实都是红色
        board_temp[sidex][sidey] = self.RED
        # 每次落完边，更新领地
        board_temp = self.update_territory(board_temp, (sidex, sidey), self.RED)
        # 落完边之后的领地数量
        number_of_territory_after = self.get_number_of_territory(board_temp, self.RED)
        # 如果领地数量变多 那么就意味着这 一方 的玩家需要再走一条边【Flag标识】
        flag = True if number_of_territory_after > number_of_territory_before else False
        # 如果棋盘产生了新的领地，那么就不翻转棋盘，依然是这个棋盘
        # 也就是说 红色依然需要再走一条边
        if flag is False:
            board_temp = self.change_perspectives(board_temp, self.BLUE)
            # 返回flag 以及一维的棋盘
            return False, board_temp.reshape(-1)
        # 返回flag 以及一维的棋盘
        return True, board_temp.reshape(-1)

    # 打印棋盘
    def print_dots_and_boxes_board(self, board):
        board = board.copy()
        board = board.reshape(self.board_size, self.board_size)
        dic = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
        print('\033[1;30;45m%s\033[0m' % '  A B C D E F G H I J K   ')
        print('\033[1;30;45m%s\033[0m' % 'A ', end="")
        # print(1, end="")
        for i in range(self.board_size):
            for j in range(self.board_size):
                # 领地
                if (i + 1) % 2 == 0 and (j + 1) % 2 == 0:
                    if board[i][j] == self.BLUE_territory:
                        print('\033[1;34m%s\033[0m' % 'B', end=" ")
                    elif board[i][j] == self.RED_territory:
                        print('\033[1;31m%s\033[0m' % 'R', end=" ")
                    else:
                        print('\033[1m%s\033[0m' % 'E', end=" ")
                # 点
                elif (i + 1) % 2 != 0 and (j + 1) % 2 != 0:
                    print('\033[1;33m%s\033[0m' % '⊙', end="")  # 显示方式：1； 　字体色：31；  背景色：46
                # 横边
                elif (i + 1) % 2 != 0 and (j + 1) % 2 == 0:
                    if board[i][j] == self.BLUE:
                        print('\033[1;34m%s\033[0m' % '———', end="")
                    elif board[i][j] == self.RED:
                        print('\033[1;31m%s\033[0m' % '———', end="")
                    else:
                        print("   ", end="")
                # 竖边
                elif (i + 1) % 2 == 0 and (j + 1) % 2 != 0:
                    if board[i][j] == self.BLUE:
                        print('\033[1;34m%s\033[0m' % '│', end=" ")
                    elif board[i][j] == self.RED:
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
            if i + 2 < self.board_size:
                j = i + 1
                j = dic[j] + " "
                print('\033[1;30;45m%s\033[0m' % j, end="")
            elif i + 2 == self.board_size - 1:
                print('\033[1;30;45m%s\033[0m' % 'K ', end="")
            elif i + 2 == self.board_size:
                print('\033[1;30;45m%s\033[0m' % 'K ', end="")
        print('\033[1;30;45m%s\033[0m' % '  A B C D E F G H I J K   ')

    # 更新领地
    def update_territory(self, board, side, player):
        # 获取到该边所有的领域
        # 判断边的上下左右是否是领域，如果是领域，那么这个边就是属于临近的领域
        sidex = side[0]
        sidey = side[1]
        # 浅拷贝一下棋盘
        board_temp = board.copy()
        board_temp = board_temp.reshape(self.board_size, self.board_size)
        # 有可能不需要加，可能存在调用了 get_next_state 了，边已经加上了
        # 后来思考 还是需要添加 因为调用的地方不止 get-next-state 一个地方
        board_temp[sidex][sidey] = player
        # 临近的领域列表
        adjacent_territory = []
        # 接下来就统计该边的上下左右的四个方向上 的领地数量
        # 上
        if (sidex + 1, sidey) in self.all_sites_2D:
            adjacent_territory.append((sidex + 1, sidey))
        # 下
        if (sidex - 1, sidey) in self.all_sites_2D:
            adjacent_territory.append((sidex - 1, sidey))
        # 左
        if (sidex, sidey - 1) in self.all_sites_2D:
            adjacent_territory.append((sidex, sidey - 1))
        # 右
        if (sidex, sidey + 1) in self.all_sites_2D:
            adjacent_territory.append((sidex, sidey + 1))
        # [(x1,y1),(x2,y2),(x3,y3)......]
        # 接着对相邻的领地进行判断
        for i in range(len(adjacent_territory)):
            # 获取领地的横纵坐标
            sitex = adjacent_territory[i][0]
            sitey = adjacent_territory[i][1]
            # sitex \ sitey 分别是 site 的二维坐标(检查该site的四条边是否都已经被占用)
            # 如果这条边落完之后 它的相邻领地的四条边都不是空 那么就意味着 这条边是该领地的最后一条边
            if board_temp[sitex - 1][sitey] != self.EMPTY \
                    and board_temp[sitex + 1][sitey] != self.EMPTY \
                    and board_temp[sitex][sitey - 1] != self.EMPTY \
                    and board_temp[sitex][sitey + 1] != self.EMPTY:
                # 根据玩家 给棋盘附上对应的值
                if player == self.RED:
                    board_temp[sitex][sitey] = self.RED_territory
                elif player == self.BLUE:
                    board_temp[sitex][sitey] = self.BLUE_territory
        return board_temp

    # 获取棋盘上当前玩家的领地的数量
    def get_number_of_territory(self, board, player):
        # 用来判断是否右新的领域被占时使用：
        # 落边之前对领地数目进行统计，落了一条边之后再进行统计，如果后者比前者大，则证明有领地被占据了
        count = 0
        # 浅拷贝一下原始棋盘
        board_temp = board.copy()
        board_temp = board_temp.reshape(self.board_size, self.board_size)
        for i in range(self.board_size):
            for j in range(self.board_size):
                # 同一方的领地是同号的，所以相乘大于零
                # i j 满足关系 用来判断领地的数量 and 后面的判断用来判断是不是player的领地
                if (i + 1) % 2 == 0 and (j + 1) % 2 == 0 and board_temp[i][j] * player > 0:
                    count += 1
        return count

    # 判断当前棋盘上的某条边 被走了之后会不会占据领地
    def Whether_or_not_occupy_territory(self, board, sides):
        # 传入的待判断的sides是一维的
        sides_x = self.all_sides_2D[self.all_sides_1D.index(sides)][0]
        sides_y = self.all_sides_2D[self.all_sides_1D.index(sides)][1]
        # 浅拷贝一下原始棋盘
        board_temp = board.copy()
        board_temp = board_temp.reshape(self.board_size, self.board_size)
        # 先获取未走此条边的领地数目【统计使用 所以可以是任意的玩家】
        update_territory_before = self.get_number_of_territory(board_temp, self.RED)
        # 更新领地【统计使用 所以可以是任意的玩家】
        board_temp = self.update_territory(board_temp, (sides_x, sides_y), self.RED)
        # 再获取更新领地之后的边数 【统计使用 所以可以是任意的玩家】
        update_territory_after = self.get_number_of_territory(board_temp, self.RED)
        if update_territory_after > update_territory_before:
            return True
        return False

    # 对弈结束之后 获取赢家是谁
    def get_winner(self, board):
        territory_of_red = 0
        territory_of_blue = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                # 领地
                if (i + 1) % 2 == 0 and (j + 1) % 2 == 0:
                    if board[i][j] == self.RED_territory:
                        territory_of_red += 1
                    if board[i][j] == self.BLUE_territory:
                        territory_of_blue += 1
        print("|" * 100)
        print("红色方的领地数目为：", territory_of_red)
        print("蓝色方的领地数目为：", territory_of_blue)
        print("|" * 100)
        if territory_of_blue > territory_of_red:
            return self.BLUE
        else:
            return self.RED

    """
        PK部分代码
    """

    # pk on terminal
    def get_human_input(self, primary_board):
        alpha_digit_dic = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10}
        flag = "N"
        print("=" * 10 + " 回退到上一步，请输入 K K " + "=" * 10)
        # 循环输入
        while flag == "N" or flag == "n":
            board = primary_board.copy()
            board = board.reshape(-1)
            # 横坐标
            action_x = input("请输入起始点的 二维坐标 横坐标:")
            while not action_x.isalpha():
                action_x = input("输入坐标有误，请重新输入起始点的二维坐标横坐标:")
            action_x = action_x.upper()
            action_x = alpha_digit_dic[action_x]
            # 纵坐标
            action_y = input("请输入起始点的 二维坐标 纵坐标:")
            while not action_y.isalpha():
                action_y = input("输入坐标有误，请重新输入起始点的二维坐标纵坐标:")
            action_y = action_y.upper()
            action_y = alpha_digit_dic[action_y]
            next_action = action_x * self.board_size + action_y
            # 尝试加一点容错机制
            # 标识回退一步
            if next_action == 120:
                return 1000
            if next_action not in self.all_sides_1D:
                print("坐标有误，该条边根本不存在！！！")
                continue
            elif board[next_action] != self.EMPTY:
                print("坐标有误，该条边已经走过了！！！")
                continue
            # 标识待走的边
            board[next_action] = 10
            print("|" * 50)
            print("|" * 50)
            print("打印下一个棋盘落边位置状态预览：")
            self.print_temp_board(board)
            print("|" * 50)
            print("|" * 50)
            flag = input("确认输入正确，请输入 Y, 想重新输入点位，请输入 N:")
        return next_action

    def print_temp_board(self, board):
        board = board.copy()
        board = board.reshape(self.board_size, self.board_size)
        dic = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
        print('\033[1;30;45m%s\033[0m' % '  A B C D E F G H I J K   ')
        print('\033[1;30;45m%s\033[0m' % 'A ', end="")
        for i in range(self.board_size):
            for j in range(self.board_size):
                # 领地
                if (i + 1) % 2 == 0 and (j + 1) % 2 == 0:
                    if board[i][j] == self.BLUE_territory:
                        print('\033[1;34m%s\033[0m' % 'B', end=" ")
                    elif board[i][j] == self.RED_territory:
                        print('\033[1;31m%s\033[0m' % 'R', end=" ")
                    else:
                        print('\033[1m%s\033[0m' % 'E', end=" ")
                # 点
                elif (i + 1) % 2 != 0 and (j + 1) % 2 != 0:
                    print('\033[1;33m%s\033[0m' % '⊙', end="")  # 显示方式：1； 　字体色：31；  背景色：46
                # 横边
                elif (i + 1) % 2 != 0 and (j + 1) % 2 == 0:
                    if board[i][j] == self.BLUE:
                        print('\033[1;34m%s\033[0m' % '———', end="")
                    elif board[i][j] == self.RED:
                        print('\033[1;31m%s\033[0m' % '———', end="")
                    elif board[i][j] == 10:
                        print('\033[1;33m%s\033[0m' % '———', end="")
                    else:
                        print("   ", end="")
                # 竖边
                elif (i + 1) % 2 == 0 and (j + 1) % 2 != 0:
                    if board[i][j] == self.BLUE:
                        print('\033[1;34m%s\033[0m' % '│', end=" ")
                    elif board[i][j] == self.RED:
                        print('\033[1;31m%s\033[0m' % '│', end=" ")
                    elif board[i][j] == 10:
                        print('\033[1;33m%s\033[0m' % '│', end=" ")
                    else:
                        print(" ", end=" ")
            if (i + 1) % 2 == 1:
                print(" ", end="")
            k = dic[i]
            k = k + " "
            # print('\033[1;30;45m%s\033[0m' % '  ', end="")
            print('\033[1;30;45m%s\033[0m' % k, end="")
            print("\n", end="")
            if i + 2 < self.board_size:
                j = i + 1
                j = dic[j] + " "
                print('\033[1;30;45m%s\033[0m' % j, end="")
            elif i + 2 == self.board_size - 1:
                print('\033[1;30;45m%s\033[0m' % 'K ', end="")
            elif i + 2 == self.board_size:
                print('\033[1;30;45m%s\033[0m' % 'K ', end="")
        print('\033[1;30;45m%s\033[0m' % '  A B C D E F G H I J K   ')

    def get_territory_for_PK(self, board):
        territory_of_red = 0
        territory_of_blue = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                # 领地
                if (i + 1) % 2 == 0 and (j + 1) % 2 == 0:
                    if board[i][j] == self.RED_territory:
                        territory_of_red += 1
                    if board[i][j] == self.BLUE_territory:
                        territory_of_blue += 1
        return territory_of_red, territory_of_blue

    # 针对PK部分写的game-over
    def game_over(self, board, player=None):
        board_temp = board.copy()
        board_temp = board_temp.reshape(self.board_size, self.board_size)
        territory_of_blue = 0
        territory_of_red = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i + 1) % 2 == 0 and (j + 1) % 2 == 0:
                    if board_temp[i][j] == self.RED:
                        territory_of_red += 1
                    if board_temp[i][j] == self.BLUE:
                        territory_of_blue += 1
                    if board_temp[i][j] == self.EMPTY:
                        return self.GAME_NOT_END
        # 上面没有返回
        return self.WIN if territory_of_red > territory_of_blue else self.LOSE

    # 满足抽象方法的部分（无实际作用）
    def get_legal_action(self, board, layers, start_pos=None):
        pass
