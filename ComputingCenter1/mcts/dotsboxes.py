from mcts.abstract_mcts import AbstractMcts, AbstractNode
from collections import namedtuple
import numpy as np
import queue
from Storage import Log
from endgame import Endgame


class Mcts(AbstractMcts):

    def __init__(self, game: object, args: object, trainer: object, replay_buffer: object):
        super().__init__(game, args, trainer, replay_buffer)
        self.game = game
        self.args = args
        self.trainer = trainer
        self.replay_buffer = replay_buffer
        # Node information, storing p, v
        self.node_info = dict()
        self.namedtuple = namedtuple('node_info', ['pi', 'v'])
        self.root = None

    def select_action_dots(self, proc_id: int, board, num_iter: int, layers: int, policy: str, last_move=None):
        _, v = self.trainer.predict(board, self.trainer.net_work)
        print("network predict V: ", v)
        return_data, trajectory = [], []
        flag = 0
        if self.args.start_search_step_threshold <= layers < self.args.start_search_step_threshold + self.args.strong_search_step_interval:
            # 2000 search
            self.args.train_num_search, self.args.strong_search_num = self.args.strong_search_num, self.args.train_num_search
            flag = 1
        if last_move is None:  # 第一步
            leaf_endgame_num, leaf_end_num, leaf_num, total_endgame, hash_num, predict_num = self.search(num_iter, board, layers, -1)
        else:
            leaf_endgame_num, leaf_end_num, leaf_num, total_endgame, hash_num, predict_num = self.search(num_iter, board, layers, last_move)
            if leaf_num + leaf_end_num + leaf_endgame_num + total_endgame == -4:   # 可以快速走未进行搜索
                if flag: # 2000 search
                    self.args.train_num_search, self.args.strong_search_num = self.args.strong_search_num, self.args.train_num_search
                print(self.root.root_quick_move)
                return self.root.root_quick_move, None, None, True
        if self.args.train_num_search == 0:
            if flag:
                # 2000 search
                flag = 0
                self.args.train_num_search, self.args.strong_search_num = self.args.strong_search_num, self.args.train_num_search
            return None, None, None, False
        print("根节点的Q值: ", self.root.my_q)
        action, pi = self.get_action_by_policy(self.root.child_N, policy)
        data = self.extract_data(self.root, num_iter)
        return_data.extend(data)
        key = self.game.to_string(board)
        if key in self.node_info:
            trajectory.append((layers, board, self.node_info[key].pi, np.array(pi, dtype=float),
                               self.node_info[key].v, self.root.my_q))
        if proc_id == self.args.SingleProcess or proc_id == self.args.print_log_process:
            """
                确保打印的Q值的正确性
                1.Log.print_simulation_results里的有关于Q的内容都取反。
                2.如果child里对应的边可以成，那就再取一个反。
            """
            for i in range(len(self.root.child_N)):
                if self.root.child_N[i] != 0:
                    if self.game.Whether_or_not_occupy_territory(board, i) is True:
                        self.root.child_Q[i] = -self.root.child_Q[i]
            print("叶子节点数目为: ", leaf_num, "     终局节点数目为: ", leaf_end_num, "     残局节点数目为: ", leaf_endgame_num)
            print("叶子节点中的残局命中率: ", round(leaf_endgame_num / leaf_num, 2))
            print("所有节点中的残局命中率: ", round(total_endgame / self.args.train_num_search, 2))
            print("hash node num: ", hash_num, "     net_work node num: ", predict_num, "     hash hit rate: ", round(hash_num / (hash_num + predict_num), 2))
            Log.print_simulation_results("Choose Side: ", dict(
                zip(np.nonzero(self.root.child_N)[0], list(self.root.child_N[np.nonzero(self.root.child_N)[0]]))),
                                         dict(zip(np.nonzero(self.root.child_N)[0], [round(i, 3) for i in -self.root.child_Q[np.nonzero(self.root.child_N)[0]]])),
                                         dict(zip(np.nonzero(self.root.child_N)[0], [round(i, 3) for i in (
                                                 self.game.args.Cpuct * self.root.child_Pi * np.sqrt(
                                             self.root.my_n) / (1 + self.root.child_N))[
                                             np.nonzero(self.root.child_N)[0]]])))
        if flag:
            # 2000 search
            flag = 0
            self.args.train_num_search, self.args.strong_search_num = self.args.strong_search_num, self.args.train_num_search
        return action, return_data, trajectory, False

    def search(self, num_iter, board, layers, action=-1):
        leaf_num, leaf_endgame_num, leaf_end, total_endgame, hash_num, predict_num = 0, 0, 0, 0, 0, 0
        parent = Node(self.game, board, move=action, layers=layers, is_quick=False, parent=None, flag=False)
        # Root and root's children is reinitialized when 'search' is called every time
        if action == -1:
            self.root = Node(self.game, board, move=action, layers=layers, is_quick=False, parent=parent, flag=True)
        else:
            self.root = Node(self.game, board, move=action, layers=layers,  is_quick=True, parent=parent, flag=True)
            if len(self.root.root_quick_move) != 0:  # root可以快速走，不需要搜索
                assert self.args.use_chain_loop_prune
                return -1, -1, -1, -1, 0, 0
        # Use MCTS method to search train_num_search times
        print("总结点数：", self.args.train_num_search)
        for _ in range(self.args.train_num_search):
            # expand
            leaf = self.root.expand_child(num_iter)
            if leaf.is_endgame:
                total_endgame += 1
            if leaf.my_n == 0:  # 新扩展的叶子节点
                leaf_num += 1
                if len(leaf.parent.child) == 1:  # 父节点被扩展
                    leaf_num -= 1
                if leaf.is_endgame:
                    leaf_endgame_num += 1
                if not leaf.is_endgame and leaf.is_end:
                    leaf_end += 1
            # reach leaf
            # 这个地方返回的is_end的值是正确的
            if leaf.is_end != self.game.GAME_NOT_END:
                pi = None
                # 点格棋的这个v值需要根据情况判断的
                v = np.array(leaf.is_end)
            # don't reach leaf, using the network to predict
            else:
                key = self.game.to_string(leaf.board)
                if key not in self.node_info:
                    pi, v = self.trainer.predict(leaf.board.reshape(self.game.board_size, self.game.board_size),
                                                 self.trainer.net_work)
                    self.node_info[key] = self.namedtuple._make([pi, v])
                    predict_num += 1
                else:
                    pi, v = self.node_info[key].pi, self.node_info[key].v
                    hash_num += 1
            leaf.expand_myself(pi)
            # backup value v
            leaf.backup(v)
        return leaf_endgame_num, leaf_end, leaf_num, total_endgame, hash_num, predict_num

    def extract_data(self, node, num_iter):
        return_data = []
        if not node:
            return []
        children = queue.Queue()
        children.put(node)
        while len(children.queue) > 0:
            current = children.get()
            # # extract data
            pi = current.child_N / np.sum(current.child_N)
            for b, p in self.game.board_flip(current.board, pi):
                return_data.append(
                    [num_iter, current.layers, current.my_n, [b, np.array(p, dtype=np.float64), current.my_q]])
            if not self.args.data_augmentation:
                return return_data
            # explore down
            for key, child in current.child.items():
                n_threshold = int(self.args.N_threshold)
                if np.sum(child.child_N) > 1e-5 and \
                        (child.my_n > 0.8 * self.args.N_threshold_num[0] \
                         or (child.my_n > self.args.N_threshold_num[1] and abs(child.my_q) > self.args.N_Q_threshold[0]) # 0.8
                         or (child.layers < 25 and (child.my_n > self.args.N_threshold_num[1] and abs(child.my_q) > self.args.N_Q_threshold[1]))
                         or (25 <= child.layers < 28 and (child.my_n > self.args.N_threshold_num[2] and abs(child.my_q) > self.args.N_Q_threshold[2]))
                         or (28 <= child.layers <= 32 and (child.my_n > self.args.N_threshold_num[3] and abs(child.my_q) > self.args.N_Q_threshold[3]))
                         or (child.layers >= 33 and (child.my_n > self.args.N_threshold_num[4] and abs(child.my_q) > self.args.N_Q_threshold[4]))
                        ):
                    children.put(child)
        return return_data

    def get_action_by_policy(self, counts, policy):
        pi = ((counts) / np.sum(counts))
        if policy == "greedy policy":
            action = np.argmax(pi)
        elif policy == "sharpening policy":
            p = (counts ** self.args.sharpening_policy_t) / np.sum(counts ** self.args.sharpening_policy_t)
            action = np.random.choice(len(p), p=p)
        else:
            action = np.random.choice(len(pi), p=pi)
            assert 0
        return action, pi

    # 满足抽象方法需要，无实际作用
    def select_action(self, proc, board, num_iter, layers, policy):
        pass


class Node(AbstractNode):

    def __init__(self, game, board, move, layers, is_quick=False, parent=None, flag=False):
        super().__init__(game, board, move, layers, parent, flag)
        self.use_quick = is_quick  # root 的父节点不用analyse board
        self.game = game
        self.board = board
        self.action = move
        self.parent = parent
        self.is_expanded = False
        self.child = dict()
        self.child_Pi = np.zeros([game.action_size], dtype='float32')  # Neural network predicts probability
        self.child_Q = np.zeros([game.action_size], dtype='float32')  # Mean posterior reward
        self.child_N = np.zeros([game.action_size], dtype=int)
        self.layers = layers
        # 每一次都是站在红方的视角判断输赢，如果赢了，那就是返回1，如果输了难就返回-1
        self.is_endgame = False
        self.new_territory_flag = flag
        self.epsilon = 5e-3
        # 初始化为-1，假定父子结点属于对手方
        self.child_select = -1 * np.ones([game.action_size], dtype=int)
        # control and give up control 两个动作，用于下次再搜到这个节点的时候扩展孩子用
        self.logical_action = []
        self.root_quick_move = []  # 根节点被open可以直接返回值，不需要search
        self.dfs_move = []  # debug 用
        self.fake_last_move = None
        assert self.game.args.CenterName == "Control Center"
        if self.game.args.use_chain_loop_prune and self.use_quick and layers != 0:
            self.analyse_board()
        self.is_end = self.game_over_dots(self.board)

    def count_box_side(self, x, y, board):
        temp_board = board.copy()
        dx = [0, -1, 0, 1]
        dy = [-1, 0, 1, 0]
        cnt = 0
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if temp_board[nx][ny] != 0:
                cnt += 1
        return cnt

    def dfs(self, x, y, vis, board):
        move, left_move, right_move = [], [], []
        vis[x][y] = 1
        a, b = self.game.judge_side_type(x, y, board)
        temp_board = board.copy()
        temp_board[x][y], vis[x][y], flag = 0, 1, 0
        # 竖边的邻边
        sx = [1, 0, -1, -1, 0, 1]
        sy = [-1, -2, -1, 1, 2, 1]
        # 横边的临边
        hx = [-1, -2, -1, 1, 2, 1]
        hy = [-1, 0, 1, 1, 0, -1]
        is_vertical = True if x % 2 == 1 else False
        for i in range(6):
            if is_vertical:
                nx, ny = x + sx[i], y + sy[i]
                if (nx, ny) in self.game.all_sides_2D and temp_board[nx][ny] == 0 and vis[nx][ny] == 0:
                    box_side = self.count_box_side(x, y - 1, temp_board) if i <= 2 else self.count_box_side(x, y + 1,
                                                                                                            temp_board)
                    if box_side == 2:
                        if flag == 0:
                            flag = 1
                            left_move.extend(self.dfs(nx, ny, vis, temp_board))
                        else:
                            right_move.extend(self.dfs(nx, ny, vis, temp_board))
            else:
                nx, ny = x + hx[i], y + hy[i]
                if (nx, ny) in self.game.all_sides_2D and temp_board[nx][ny] == 0 and vis[nx][ny] == 0:
                    box_side = self.count_box_side(x - 1, y, temp_board) if i <= 2 else self.count_box_side(x + 1, y,
                                                                                                            temp_board)
                    if box_side == 2:
                        if flag == 0:
                            flag = 1
                            left_move.extend(self.dfs(nx, ny, vis, temp_board))
                        else:
                            right_move.extend(self.dfs(nx, ny, vis, temp_board))
        if board[x][y] != 0:
            move.extend(list(reversed(left_move)) + [(x, y)] + right_move)
            # print(move)
            return move
        else:
            assert len(left_move) == 0 or len(right_move) == 0
            move.extend([(x, y)] + left_move + right_move)
            return move

    def analyse_board(self):
        temp_board = self.board.copy()
        temp_board = temp_board.reshape(self.game.board_size, self.game.board_size)
        vis = np.zeros((self.game.board_size, self.game.board_size))
        # 竖边的邻边
        sx = [1, 0, -1, -1, 0, 1]
        sy = [-1, -2, -1, 1, 2, 1]
        # 横边的临边
        hx = [-1, -2, -1, 1, 2, 1]
        hy = [-1, 0, 1, 1, 0, -1]
        x, y = self.action // self.game.board_size, self.action % self.game.board_size
        assert (x + y) % 2 == 1
        # 获取刚走过的边的type
        a, b = self.game.judge_side_type(x, y, self.board)
        side_44 = []

        # 判断action是否造成了44
        for i in range(6):
            if x % 2 == 0:
                if (x + hx[i], y + hy[i]) in self.game.all_sides_2D and temp_board[x + hx[i]][y + hy[i]] == 0:
                    t1, t2 = self.game.judge_side_type(x + hx[i], y + hy[i], self.board)
                    if (t1, t2) == (4, 4):
                        side_44.append((x + hx[i], y + hy[i]))
            else:
                if (x + sx[i], y + sy[i]) in self.game.all_sides_2D and temp_board[x + sx[i]][y + sy[i]] == 0:
                    t1, t2 = self.game.judge_side_type(x + sx[i], y + sy[i], self.board)
                    if (t1, t2) == (4, 4):
                        side_44.append((x + sx[i], y + sy[i]))
        for key in side_44:  # 将44的边都走了
            temp_board = self.game.update_territory(temp_board, (key[0], key[1]), self.game.RED)
            self.root_quick_move.append(key[0]*11 + key[1])
            self.board = temp_board
        if len(side_44) != 0:
            return

        if 3 in (a, b):  # open,chain or loop走出两种情况
            move = self.dfs(x, y, vis, temp_board)
            self.dfs_move = move.copy()
            self.dfs_move.extend((x, y))
            # print(move, x, y)
            bb = temp_board.copy()
            bb[x][y] = 0  # 去掉上一条边
            type = [(self.game.judge_side_type(i[0], i[1], bb)) for i in move]
            type_34 = [i for i in type if i == (3, 4) or i == (4, 3)]
            index = move.index((x, y))
            if ((3, 4) in type or (4, 3) in type) and len(type_34) == 2:  # 只剩3个格子的loop(4loop走了一个格子),或者4个格子的control的loop
                assert type[1] == (3, 3)
                if len(move) == 3:
                    action = [i for i in move if i != (x, y)]
                    for key in action:
                        temp_board = self.game.update_territory(temp_board, (key[0], key[1]), self.game.RED)
                        self.root_quick_move.append(key[0] * self.game.board_size + key[1])
                    self.board = temp_board
                    self.layers += len(action)
                else:
                    del move[index]
                    for key in move[:-3]:
                        temp_board = self.game.update_territory(temp_board, (key[0], key[1]), self.game.RED)
                        self.root_quick_move.append(key[0] * self.game.board_size + key[1])
                    self.board = temp_board
                    self.layers += len(move[:-3])
                    # control
                    self.logical_action.append(move[-2][0] * self.game.board_size + move[-2][1])
                    # get loop
                    self.logical_action.append(move[-1][0] * self.game.board_size + move[-1][1])
                    self.logical_action.append(move[-3][0] * self.game.board_size + move[-3][1])
            elif all(element == (3, 3) for element in type):  # opened loop
                assert len(move) >= 4
                del move[index]
                action = move[:-3]
                for key in action:
                    temp_board = self.game.update_territory(temp_board, (key[0], key[1]), self.game.RED)
                    self.root_quick_move.append(key[0]*self.game.board_size+key[1])
                self.layers += len(action)
                self.board = temp_board
                is_end = self.game_over_dots(self.board)
                if is_end != self.game.GAME_NOT_END:
                    return
                # control
                self.logical_action.append(move[-2][0]*self.game.board_size + move[-2][1])
                # get loop
                self.logical_action.append(move[-1][0]*self.game.board_size + move[-1][1])
                self.logical_action.append(move[-3][0]*self.game.board_size + move[-3][1])
            else:  # opened chain
                left_move = list(reversed(move[:index]))
                right_move = move[index + 1:]
                if len(left_move) >= 2 or len(right_move) >= 2:  # 可以control
                    short_move = left_move if len(left_move) <= len(right_move) else right_move
                    long_move = left_move if len(left_move) > len(right_move) else right_move
                    for key in short_move:
                        temp_board = self.game.update_territory(temp_board, (key[0], key[1]), self.game.RED)
                        self.root_quick_move.append(key[0]*11+key[1])
                    for key in long_move[:-2]:
                        temp_board = self.game.update_territory(temp_board, (key[0], key[1]), self.game.RED)
                        self.root_quick_move.append(key[0] * 11 + key[1])
                    if len(long_move) == 2:
                        self.fake_last_move = x*self.game.board_size+y
                    self.layers += len(short_move) + len(long_move[:-2])
                    self.board = temp_board
                    is_end = self.game_over_dots(self.board)
                    if is_end != self.game.GAME_NOT_END:
                        return
                    # control
                    self.logical_action.append(long_move[-1][0] * self.game.board_size + long_move[-1][1])
                    # get chain
                    self.logical_action.append(long_move[-2][0] * self.game.board_size + long_move[-2][1])
                else:  # 格子都拿走
                    action = left_move + right_move
                    for key in action:
                        temp_board = self.game.update_territory(temp_board, (key[0], key[1]), self.game.RED)
                        self.root_quick_move.append(key[0] * 11 + key[1])
                    self.layers += len(action)
                    self.board = temp_board

    def game_over_dots(self, board):  # 训练使默认当前红色方
        board_temp = board.copy()
        board_temp = board_temp.reshape(self.game.board_size, self.game.board_size)
        territory_of_blue = 0
        territory_of_red = 0
        side_num = 0  # 占领的边的数量
        for i in range(self.game.board_size):
            for j in range(self.game.board_size):
                if (i + j) % 2 == 1 and board_temp[i][j] != 0:
                    side_num += 1
                if (i + 1) % 2 == 0 and (j + 1) % 2 == 0:
                    if board_temp[i][j] == self.game.RED_territory:
                        territory_of_red += 1
                    if board_temp[i][j] == self.game.BLUE_territory:
                        territory_of_blue += 1
                    if territory_of_red > (self.game.site_size // 2):
                        return self.game.WIN
                    if territory_of_blue > (self.game.site_size // 2):
                        return self.game.LOSE
        if self.game.args.use_endgame and side_num >= self.game.args.side_size // 2 and self.parent is not None and self.parent.parent is not None:  # 根节点不用这个

            if self.game.pd_unopened_game(board_temp):
                opt = Endgame(board)
                assert opt.get_structure()
                for key in opt.G_loops:
                    if key.length % 2 != 0 or key.length < 4:
                        return self.game.GAME_NOT_END
                self.is_endgame = True
                v = opt.calculate_v(opt.G_chains, opt.G_loops)
                if territory_of_red > territory_of_blue + v:
                    return self.game.WIN
                elif territory_of_red < territory_of_blue + v:
                    return self.game.LOSE
                else:
                    assert 0
        # 上面没有返回，证明还没结束
        return self.game.GAME_NOT_END

    @property
    def my_n(self):
        return self.parent.child_N[self.action]

    @my_n.setter
    def my_n(self, num):
        self.parent.child_N[self.action] = num

    @property
    def my_q(self):
        return self.parent.child_Q[self.action]

    @my_q.setter
    def my_q(self, value):
        self.parent.child_Q[self.action] = value

    def select_child_by_uct(self, num_iter):
        if self.game.args.use_chain_loop_prune and len(self.logical_action) != 0: # 存在必在2个或三个动作中选
            assert len(self.logical_action) == 2 or len(self.logical_action) == 3
            actions = self.logical_action
        else:
            actions = self.game.get_legal_action_dots(self.board)
            assert actions is not None
        # 将Pi归一化,和为1，非负
        # pi = (self.child_Pi[actions]+self.epsilon) / sum(self.child_Pi[actions]+self.epsilon)
        pi = self.child_Pi[actions] / sum(self.child_Pi[actions]) if sum(self.child_Pi[actions]) > 1e-5 else self.child_Pi[actions]
        q = self.child_Q * self.child_select
        q = q[actions]
        # 获得U函数
        u = self.game.args.Cpuct * pi * np.sqrt(self.my_n) / (1 + self.child_N[actions])
        # UCT公式
        uct = q + u
        assert len(uct) > 0
        return actions[np.argmax(uct)]

    def expand_child(self, num_iter) -> object:
        current = self
        while current.is_expanded and current.is_end == self.game.GAME_NOT_END:
            best_move = current.select_child_by_uct(num_iter)
            if best_move not in current.child:
                # flag标识是否有占有领地情况，如果没有占有领地情况，那么new_board也是没有经过翻转的board
                new_territory_flag, new_board = self.game.get_next_board(current.board, best_move)
                # 如果flag为True，那么证明当前的best_move产生了领地，那么这个点就需要特殊标注一下
                current.child[best_move] = Node(self.game, new_board, best_move, current.layers + 1, True, current,
                                                new_territory_flag)
                if new_territory_flag is True:
                    current.child_select[best_move] = 1  # 父子结点属于同一方
            current = current.child[best_move]
        return current

    def expand_myself(self, child_pi):
        self.is_expanded = True
        self.child_Pi = child_pi

    def backup(self, value):
        """
            current 是 当前的节点
            child 是 当前的节点的孩子节点
        """
        # 先更新叶子节点
        current = self
        flag = 0
        child = None
        # 退出循环的时候 current是指向root的parent节点的 child是指向root节点的
        while current.parent is not None:
            current.my_n += 1
            if flag == 0:
                # 更新新创建的叶子节点
                flag = 1
                current.my_q = ((current.my_n - 1) * current.my_q + value) / current.my_n
            else:
                # 如果产生了新的领地，v值不变号
                if child.new_territory_flag:
                    value = value
                    child.my_q = ((child.my_n - 1) * child.my_q + value) / child.my_n
                # 先修改q值 再修改value值
                elif current.new_territory_flag is True and child.new_territory_flag is False:
                    child.my_q = ((child.my_n - 1) * child.my_q + value) / child.my_n
                    value = value * (-1)
                elif current.new_territory_flag is False and child.new_territory_flag is False:
                    child.my_q = ((child.my_n - 1) * child.my_q + value) / child.my_n
                    value = value * (-1)
            child = current
            current = current.parent
        # 更新root节点
        else:
            child.my_q = ((child.my_n - 1) * child.my_q + value) / child.my_n