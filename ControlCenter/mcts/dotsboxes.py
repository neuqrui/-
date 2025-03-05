from mcts.abstract_mcts import AbstractMcts, AbstractNode
from collections import namedtuple
import numpy as np
import queue
from Storage import Log


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

    def select_action_dots(self, proc_id: int, board: list, num_iter: int, layers: int, policy: str):
        return_data, trajectory = [], []
        self.search(num_iter, board, layers)
        action, pi = self.get_action_by_policy(self.root.child_N, policy)
        return_data.extend(self.extract_data(self.root, num_iter))
        trajectory.append((layers, board, self.node_info[self.game.to_string(board)].pi, np.array(pi, dtype=np.float64),
                           self.node_info[self.game.to_string(board)].v, self.root.my_q))
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
            Log.print_simulation_results("Choose Side: ", dict(zip(np.nonzero(self.root.child_N)[0], list(self.root.child_N[np.nonzero(self.root.child_N)[0]]))),
                                         dict(zip(np.nonzero(self.root.child_N)[0], [-round(i, 3) for i in self.root.child_Q[np.nonzero(self.root.child_N)[0]]])),
                                         dict(zip(np.nonzero(self.root.child_N)[0], [round(i, 3) for i in (self.game.args.Cpuct * self.root.child_Pi * np.sqrt(self.root.my_n) / (1 + self.root.child_N))[np.nonzero(self.root.child_N)[0]]])))
        return action, return_data, trajectory

    def search(self, num_iter, board, layers, action=-1):
        parent = Node(self.game, board, move=action, layers=layers, parent=None, flag=False)
        # Root and root's children is reinitialized when 'search' is called every time
        self.root = Node(self.game, board, move=action, layers=layers, parent=parent, flag=False)
        # Use MCTS method to search train_num_search times
        for _ in range(self.args.train_num_search):
            # expand
            leaf = self.root.expand_child(num_iter)
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
                    pi, v = self.trainer.predict(leaf.board.reshape(self.game.board_size, self.game.board_size), self.trainer.net_work)
                    self.node_info[key] = self.namedtuple._make([pi, v])
                else:
                    pi, v = self.node_info[key].pi, self.node_info[key].v
            leaf.expand_myself(pi)
            # backup value v
            leaf.backup(v)

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
                break
            # explore down
            for key, child in current.child.items():
                n_threshold = int(self.args.N_threshold)
                if np.sum(child.child_N) > 1e-5 and \
                        (child.my_n > self.args.N_threshold \
                         or (child.my_n > 0.6 * n_threshold and abs(child.my_q) > self.args.N_Q_threshold)
                         or (child.my_n > 0.1 * n_threshold and abs(child.my_q) > self.args.N_Q_threshold2)):
                    children.put(child)
        return return_data

    def get_action_by_policy(self, counts, policy):
        pi = counts / np.sum(counts)
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

    def __init__(self, game, board, move, layers, parent=None, flag=False):
        super().__init__(game, board, move, layers, parent, flag)
        self.game = game
        self.board = board
        self.action = move
        self.parent = parent
        self.is_expanded = False
        self.child = dict()
        self.child_Pi = np.zeros([game.action_size], dtype=np.float32)  # Neural network predicts probability
        self.child_Q = np.zeros([game.action_size], dtype=np.float32)  # Mean posterior reward
        self.child_N = np.zeros([game.action_size], dtype=np.int32)
        self.layers = layers
        # 每一次都是站在红方的视角判断输赢，如果赢了，那就是返回1，如果输了难就返回-1
        self.is_end = self.game.game_over_dots(self.board)
        self.new_territory_flag = flag
        # 初始化为-1，假定父子结点属于对手方
        self.child_select = -1 * np.ones([game.action_size], dtype=np.int32)

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
        actions = self.game.get_legal_action_dots(self.board)
        # 将Pi归一化,和为1，非负
        pi = self.child_Pi[actions] / sum(self.child_Pi[actions]) if sum(self.child_Pi[actions]) > 1e-5 else self.child_Pi[actions]
        # 获得Q函数，并最小最大归一化
        q = -self.child_Q
        """
            修改思路：
                不应该下面一层的所有子节点的Q值都取反，而是
                未能生成新领地的action对应的Q值取反，
                能生成新的领地的action对应的Q值不变
        """
        for side in actions:
            if self.game.Whether_or_not_occupy_territory(self.board, side) is True:
                q[side] = -q[side]
        q = q[actions]
        # 获得U函数
        u = self.game.args.Cpuct * pi * np.sqrt(self.my_n) / (1 + self.child_N[actions])
        # UCT公式
        uct = q + u
        return actions[np.argmax(uct)]

    def expand_child(self, num_iter) -> object:
        current = self
        while current.is_expanded and current.is_end == self.game.GAME_NOT_END:
            best_move = current.select_child_by_uct(num_iter)
            if best_move not in current.child:
                # flag标识是否有占有领地情况，如果没有占有领地情况，那么new_board也是没有经过翻转的board
                new_territory_flag, new_board = self.game.get_next_board(current.board, best_move)
                # 如果flag为True，那么证明当前的best_move产生了领地，那么这个点就需要特殊标注一下
                current.child[best_move] = Node(self.game, new_board, best_move, current.layers + 1, current, new_territory_flag)
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
            判断是否翻转的证据：
                current 是 T 同时 child 为 F 那么就先修改值 再 对 value值取反
                以及 current 是 F 同时 child为 F 那么就
        """
        # 先更新叶子节点
        current = self
        flag = 0
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