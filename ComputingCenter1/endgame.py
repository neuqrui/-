import math
from collections import namedtuple
from game_rules.dotsboxes import Game
from game_args.dotsboxes_6 import Args
import numpy as np

Chain = namedtuple("Chain", ['length', 'move'])
Loop = namedtuple("Loop", ['length', 'move'])
Abstract_chain = namedtuple("Abstract_chain", ['length', 'state'])
Abstract_loop = namedtuple("Abstract_loop", ['length', 'state'])
dx = [0, -1, 0, 1]
dy = [-1, 0, 1, 0]


class Endgame:
    def __init__(self, board=None, last_move=None):  # control 和 opened必须给last_move
        self.board = board
        self.G_chains = []
        self.G_loops = []
        self.cnt3_side = []
        self.open_side = last_move  # last_move = (x, y)
        self.is_opened = False  # 去掉last_move，edg_cnt 都为2or4
        self.un_open = False  # edg_cnt 都为2or4
        self.is_endgame = False  # 未open之前所有格子都至少有两个边
        self.is_control = False  # 被对手control
        self.args = Args()
        self.game = Game(self.args)
        self.control_loop = False  # 被control的是loop
        self.control_chain = False  # 被control的是chain
        self.is_end_game(board)

    def change_state(self, component, v):
        assert (isinstance(component, Abstract_loop) or isinstance(component, Abstract_chain))
        if isinstance(component, Abstract_chain):
            component = Abstract_chain(component.length, v)
        elif isinstance(component, Abstract_loop):
            component = Abstract_loop(component.length, v)
        return component

    def abstract_get_child(self, board):  # child = [(board, v, move)], board = [chain, loop, chain, loop]
        # 删掉后只剩一个组件，opened最后一个要全拿，
        # board 中需要排序，因为有的情况open哪个价值都相同，这样优先输出短的，方便对照
        opened = False
        unopened = True
        control = False
        child = []
        opened_idx = -1
        control_idx = -1
        for i in range(len(board)):
            if board[i].state == 1:
                opened, unopened = True, False
                opened_idx = i
                break
            elif board[i].state == 2:
                control, unopened = True, False
                control_idx = i
                break
        if unopened:
            # 在board中选择一个open
            vis_chain = np.zeros(100)
            vis_loop = np.zeros(100)
            for i in range(len(board)):
                if isinstance(board[i], Abstract_chain) and vis_chain[board[i].length] == 1:
                    continue
                elif vis_loop[board[i].length] == 1:
                    continue
                temp_board = board.copy()
                temp_board[i] = self.change_state(temp_board[i], 1)
                child.append((temp_board, 0, temp_board[i]))
                if isinstance(temp_board[i], Abstract_chain):
                    vis_chain[temp_board[i].length] = 1
                else:
                    vis_loop[temp_board[i].length] = 1
        elif opened:
            v = 0
            if len(board) == 1:  # 只剩下一个，全拿走
                v = board[opened_idx].length
                temp_board = board.copy()
                temp_board.pop(opened_idx)
                child.append((temp_board, v, -1))
            elif board[opened_idx].length == 1:  # opened 1-chain: 获得一个格子，将此1-chain删掉，再open一个
                v = 1
                temp_board = board.copy()
                temp_board.pop(opened_idx)
                vis_chain = np.zeros(100)
                vis_loop = np.zeros(100)
                for i in range(len(temp_board)):
                    temp1_board = temp_board.copy()
                    if isinstance(temp1_board[i], Abstract_chain) and vis_chain[temp1_board[i].length] == 1:
                        continue
                    elif vis_loop[temp1_board[i].length] == 1:
                        continue
                    temp1_board[i] = self.change_state(temp1_board[i], 1)
                    child.append((temp1_board, v, -1))
                    if isinstance(temp1_board[i], Abstract_chain):
                        vis_chain[temp1_board[i].length] = 1
                    else:
                        vis_loop[temp1_board[i].length] = 1
            elif board[opened_idx].length == 2:  # opened 2-chain: 获得两个格子，将2-chain删掉，再open一个
                v = 2
                temp_board = board.copy()
                temp_board.pop(opened_idx)
                vis_chain = np.zeros(100)
                vis_loop = np.zeros(100)
                for i in range(len(temp_board)):
                    temp1_board = temp_board.copy()
                    if isinstance(temp1_board[i], Abstract_chain) and vis_chain[temp1_board[i].length] == 1:
                        continue
                    elif vis_loop[temp1_board[i].length] == 1:
                        continue
                    temp1_board[i] = self.change_state(temp1_board[i], 1)
                    child.append((temp1_board, v, -1))
                    if isinstance(temp1_board[i], Abstract_chain):
                        vis_chain[temp1_board[i].length] = 1
                    else:
                        vis_loop[temp1_board[i].length] = 1
            else:
                # keep control
                if isinstance(board[opened_idx], Abstract_chain):  # opened chain: 获得length - 2个格子并将state改为2
                    v = board[opened_idx].length - 2
                    temp_board = board.copy()
                    temp_board[opened_idx] = self.change_state(temp_board[opened_idx], 2)
                    child.append((temp_board, v, -1))
                elif isinstance(board[opened_idx], Abstract_loop):  # opened loop: 获得length - 4个格子并将state改为2
                    v = board[opened_idx].length - 4
                    temp_board = board.copy()
                    temp_board[opened_idx] = self.change_state(temp_board[opened_idx], 2)
                    child.append((temp_board, v, -1))
                # give up control
                # opened chain/loop: 获得length个格子，再将chain删掉，再open一个
                temp_board = board.copy()
                v = board[opened_idx].length
                temp_board.pop(opened_idx)
                vis_chain = np.zeros(100)
                vis_loop = np.zeros(100)
                for i in range(len(temp_board)):
                    temp1_board = temp_board.copy()
                    if isinstance(temp1_board[i], Abstract_chain) and vis_chain[temp1_board[i].length] == 1:
                        continue
                    elif vis_loop[temp1_board[i].length] == 1:
                        continue
                    temp1_board[i] = self.change_state(temp1_board[i], 1)
                    child.append((temp1_board, v, -1))
                    if isinstance(temp1_board[i], Abstract_chain):
                        vis_chain[temp1_board[i].length] = 1
                    else:
                        vis_loop[temp1_board[i].length] = 1
        elif control:
            if isinstance(board[control_idx], Abstract_chain):  # control chain: 获得两个格子，删掉chain，再open一个
                temp_board = board.copy()
                v = 2
                temp_board.pop(control_idx)
                vis_chain = np.zeros(100)
                vis_loop = np.zeros(100)
                for i in range(len(temp_board)):
                    temp1_board = temp_board.copy()
                    if isinstance(temp1_board[i], Abstract_chain) and vis_chain[temp1_board[i].length] == 1:
                        continue
                    elif vis_loop[temp1_board[i].length] == 1:
                        continue
                    temp1_board[i] = self.change_state(temp1_board[i], 1)
                    child.append((temp1_board, v, -1))
                    if isinstance(temp1_board[i], Abstract_chain):
                        vis_chain[temp1_board[i].length] = 1
                    else:
                        vis_loop[temp1_board[i].length] = 1
            elif isinstance(board[control_idx], Abstract_loop):  # control loop: 获得四个格子，删掉loop，再open一个
                temp_board = board.copy()
                v = 4
                temp_board.pop(control_idx)
                vis_chain = np.zeros(100)
                vis_loop = np.zeros(100)
                for i in range(len(temp_board)):
                    temp1_board = temp_board.copy()
                    if isinstance(temp1_board[i], Abstract_chain) and vis_chain[temp1_board[i].length] == 1:
                        continue
                    elif vis_loop[temp1_board[i].length] == 1:
                        continue
                    temp1_board[i] = self.change_state(temp1_board[i], 1)
                    child.append((temp1_board, v, -1))
                    if isinstance(temp1_board[i], Abstract_chain):
                        vis_chain[temp1_board[i].length] = 1
                    else:
                        vis_loop[temp1_board[i].length] = 1
        return child

    def get_possible_moves(self):
        assert self.is_endgame
        move = []
        # 对手open了最后一个component特例
        if not self.is_control and len(self.G_chains) + len(self.G_loops) == 1 and self.is_opened:
            opc, idx = self.find_opened_component()
            temp = []
            if isinstance(opc, Chain):
                left = list(reversed(opc.move[:idx]))
                right = opc.move[idx + 1:]
                move.append(left + right)
            elif isinstance(opc, Loop):
                left = opc.move[:idx]
                right = opc.move[idx + 1:]
                move.append(right + left)
        # 被对手control
        elif self.is_control:
            temp = self.cnt3_side.copy()
            for key in self.G_chains:
                temp_temp = temp.copy()
                temp_temp.append(key.move[0])
                move.append(temp_temp)
                if key.length == 2:  # open 2-chain 中间
                    temp_temp = temp.copy()
                    temp_temp.append(key.move[1])
                    move.append(temp_temp)
            for key in self.G_loops:
                temp_temp = temp.copy()
                temp_temp.append(key.move[0])
                move.append(temp_temp)
        # 对手open
        elif self.is_opened:
            opc, idx = self.find_opened_component()
            temp = []
            #  opened chain
            if isinstance(opc, Chain):
                if opc.length == 1 or (opc.length == 2 and idx == 1):  # 1chain 和 2chain open的中间位置
                    temp = opc.move.copy()
                    for key in self.G_chains:
                        if key == opc:
                            continue
                        temp = opc.move.copy()
                        del temp[idx]
                        temp.append(key.move[0])
                        move.append(temp)
                        if key.length == 2:  # open 2-chain 中间
                            temp = opc.move.copy()
                            del temp[idx]
                            temp.append(key.move[1])
                            move.append(temp)
                    for key in self.G_loops:
                        temp = opc.move.copy()
                        del temp[idx]
                        temp.append(key.move[0])
                        move.append(temp)
                elif opc.length == 2 and idx != 1:  # 2-chain open 边
                    temp_temp = opc.move.copy()
                    del temp_temp[idx]
                    if idx == 2:
                        temp_temp.reverse()
                    # give up control
                    for key in self.G_chains:
                        if key == opc:
                            continue
                        temp = temp_temp.copy()
                        temp.append(key.move[0])
                        move.append(temp)
                        if key.length == 2:  # open 2-chain 中间
                            temp = temp_temp.copy()
                            temp.append(key.move[1])
                            move.append(temp)
                    for key in self.G_loops:
                        temp = temp_temp.copy()
                        temp.append(key.move[0])
                        move.append(temp)
                    # keep control
                    temp = []
                    if idx == 0:
                        temp.append(opc.move[2])
                    else:
                        temp.append(opc.move[0])
                    move.append(temp)
                else:  # long_chain
                    left = list(reversed(opc.move[:idx]))
                    right = opc.move[idx + 1:]
                    remain = []
                    if len(right) >= 2:
                        temp_temp = left + right[0:-2]
                        remain = right[-2:]
                    else:
                        temp_temp = right + left[0:-2]
                        remain = left[-2:]
                    #  give up control
                    for key in self.G_chains:
                        if key == opc:
                            continue
                        temp = temp_temp + remain
                        temp.append(key.move[0])
                        move.append(temp)
                        if key.length == 2:  # open 2-chain 中间
                            temp = temp_temp + remain
                            temp.append(key.move[1])
                            move.append(temp)
                    for key in self.G_loops:
                        temp = temp_temp + remain
                        temp.append(key.move[0])
                        move.append(temp)
                    #  keep control
                    move.append(temp_temp + [remain[1]])
            elif isinstance(opc, Loop):
                left = opc.move[:idx]
                right = opc.move[idx + 1:]
                temp_temp = right + left
                temp_temp = temp_temp[:-3]
                remain = right + left
                remain = remain[-3:]
                #  give up control
                for key in self.G_chains:
                    move.append(right + left + [key.move[0]])
                    if key.length == 2:  # open 2-chain 中间
                        move.append(right + left + [key.move[1]])
                for key in self.G_loops:
                    if key == opc:
                        continue
                    move.append(right + left + [key.move[0]])
                    #  keep control
                move.append(temp_temp + [remain[1]])
        elif self.un_open:
            temp = []
            for key in self.G_chains:
                temp = [key.move[0]]
                move.append(temp)
                if key.length == 2:  # 2-chain 开中间和开两边不等效
                    temp = [key.move[1]]
                    move.append(temp)
            for key in self.G_loops:
                temp = [key.move[0]]
                move.append(temp)
        return move

    def get_board_by_move(self, move, player):  # move : [(1,2), (2, 3)]这种形式
        board = []
        temp_board = None
        before = self.game.get_number_of_territory(self.board, player)
        for key in move:
            temp_board = self.board.copy()
            last_move = None
            for i in key:
                temp_board, _ = self.game.get_next_state(temp_board, player, i[0] * self.args.board_size + i[1])
                last_move = i
            after = self.game.get_number_of_territory(temp_board, player)
            board.append((temp_board, after - before, key, last_move))
        return board

    def calculate_tb(self, chains, loops):
        assert self.is_loony_endgame(chains, loops)
        only_3chain = False
        cnt = 0
        for key in chains:
            if key.length == 3:
                cnt += 1
        if cnt == len(chains) and cnt != 0:
            only_3chain = True
        if len(chains) == 0 and len(loops) == 0:
            return 0
        elif len(chains) == 0 and len(loops) > 0:
            return 8
        elif len(loops) > 0 and only_3chain:
            return 6
        else:
            return 4

    def calculate_c(self, chains, loops):
        if not self.is_loony_endgame(chains, loops):
            print(chains, loops)
        assert self.is_loony_endgame(chains, loops)
        c = 0
        for i in chains:
            c += i.length - 4
        for j in loops:
            c += j.length - 8
        c += self.calculate_tb(chains, loops)
        return c

    def calculate_v(self, chains, loops):
        chains = sorted(chains, key=lambda point: point.length)
        loops = sorted(loops, key=lambda point: point.length)
        temp_chains = chains.copy()
        temp_loops = loops.copy()
        has_12 = False

        # 将1-chain 和 2-chain暂时删除， 因为要计算G0
        num_1chain, num_2chain = 0, 0
        del_index = []
        for i in range(len(chains)):
            if chains[i].length == 1:
                del_index.append(i)
                num_1chain += 1
            elif chains[i].length == 2:
                del_index.append(i)
                num_2chain += 1
        result_list = [chains[i] for i in range(len(chains)) if i not in del_index]
        temp_chains = result_list
        # print(temp_chains)
        if num_1chain + num_2chain != 0:
            has_12 = True
        judge = False  # 判断 G = 4 + 3 + 3
        num_3chain, num_4loop, num_6loop = 0, 0, 0
        size_G, G0_3chain = 0, 0
        G0_chains, G0_loops = self.get_G0(temp_chains, temp_loops)
        # print("G0: ", G0_chains, G0_loops, temp_chains)
        for key in G0_chains:
            if key.length == 3:
                G0_3chain += 1
        for key in temp_chains:
            size_G += key.length
            if key.length == 3:
                num_3chain += 1
        for key in temp_loops:
            size_G += key.length
            if key.length == 4:
                num_4loop += 1
            elif key.length == 6:
                num_6loop += 1
        if len(temp_loops) == 1 and num_4loop == 1 and len(temp_chains) == 2 and num_3chain == 2:
            judge = True
        v = 0
        c = self.calculate_c(temp_chains, temp_loops)
        if c >= 2:
            v = c
        elif c == 0 and num_4loop != 0 and not judge:
            v = 0
        elif num_3chain == 0 or (num_3chain == 1 and size_G % 4 == 3):
            x = self.calculate_c(G0_chains, G0_loops)
            for i in range(num_6loop):
                x = abs(x - 4) + 2
            for i in range(num_3chain - G0_3chain):
                x = x - 1
            for i in range(num_4loop):
                x = abs(x - 4)
            v = x
        else:
            if size_G % 2 == 1:
                v = 1
            if size_G % 2 == 0:
                v = 2
        if has_12:
            if num_1chain % 2 == 1 and num_2chain % 2 == 1:  # 奇奇
                v = v - 1
            elif num_1chain % 2 == 1 and num_2chain % 2 == 0:
                v = 1 - v
            elif num_1chain % 2 == 0 and num_2chain % 2 == 1:
                v = 2 - v
            elif num_1chain % 2 == 0 and num_2chain % 2 == 0:
                v = v
        return v

    def get_next_move(self):  # 终局下棋策略
        if len(self.G_chains) == 0 and len(self.G_loops) == 0:
            return None
        open_component = self.open(self.G_chains, self.G_loops)
        move = []
        if open_component.length == 2:
            open_move = open_component.move[1]
        else:
            open_move = open_component.move[0]
        if self.un_open:
            move.append(open_move)
        elif self.is_opened:
            # 拿完格子需要再open一个， 找到该open的move
            opened_component, idx = self.find_opened_component()
            remain_chains, remain_loops = self.G_chains.copy(), self.G_loops.copy()
            if opened_component in remain_chains:
                remain_chains.remove(opened_component)
            elif opened_component in remain_loops:
                remain_loops.remove(opened_component)
            open_component = self.open(remain_chains, remain_loops)
            if open_component.length == 2:
                open_move = open_component.move[1]
            else:
                open_move = open_component.move[0]

            if opened_component.length == 1:  # 1-chain被open, idx = 0 or 1
                move.extend([opened_component.move[1 - idx], open_move])
            elif opened_component.length == 2 and idx == 1:  # 2-chain 有两种形态，需要特判, |_|_|只能将两个格子都拿走，不能control
                move.extend([opened_component.move[0], opened_component.move[2], open_move])
            elif isinstance(opened_component, Chain):  # chain被open
                v_G_C = self.calculate_v(remain_chains, remain_loops)  # v(G-C) > 2, keep_control
                left = opened_component.move[:idx][::-1]
                right = opened_component.move[idx + 1:]
                if v_G_C > 2:  # keep control
                    if len(left) >= 2:
                        move.extend(left[:-2] + right + [left[-1]])
                    elif len(right) >= 2:
                        move.extend(left + right[:-2] + [right[-1]])
                else:  # give up control
                    move.extend(left + right + [open_move])
            elif isinstance(opened_component, Loop):  # loop被open
                v_G_C = self.calculate_v(remain_chains, remain_loops)  # v(G-C) > 4, keep_control
                loop_move = opened_component.move[idx + 1:] + opened_component.move[:idx]
                if v_G_C > 4:  # keep control
                    move.extend(loop_move[:-3] + [loop_move[-2]])
                else:  # give up control
                    move.extend(loop_move + [open_move])
        elif self.is_control:
            move.extend(self.cnt3_side + open_move)

        return move

    def open(self, chains, loops):
        if len(chains) == 0:
            return loops[0]
        if chains[0].length < 3:  # 有1-chain 和 2-chain， 那么open最短的
            return chains[0]
        c = self.calculate_c(chains, loops)
        chains = sorted(chains, key=lambda point: point.length)
        loops = sorted(loops, key=lambda point: point.length)
        num_3chain, num_4loop = 0, 0
        flag = False
        size_G = 0
        for key in chains:
            size_G += key.length
            if key.length == 3:
                num_3chain += 1
        for key in loops:
            size_G += key.length
            if key.length == 4:
                num_4loop += 1
        if num_4loop == 1 and num_3chain == 3 and len(chains) == 3 and len(loops) == 1:
            flag = True
        if c >= 2 and num_3chain == 1 and len(loops) > 0 and len(chains) == 1:
            return loops[0]
        elif (c == 0 or abs(c) == 1) and num_4loop != 0 and not flag:
            return loops[0]
        elif c <= -2 and num_3chain == 1 and num_4loop > 0 and (size_G - 7) % 4 == 0:
            return loops[0]
        else:
            if num_3chain != 0:
                return chains[0]
            elif len(loops) != 0:
                return loops[0]
            else:
                return chains[0]

    def find_opened_component(self):
        assert self.is_opened
        board_temp = np.array(self.board).reshape(self.args.board_size, self.args.board_size)
        flag = None
        for key in self.G_chains:
            idx = 0
            for i in key.move:
                if i == self.open_side:
                    flag = key
                    return flag, idx
                idx += 1
        for key in self.G_loops:
            idx = 0
            for i in key.move:
                if i == self.open_side:
                    flag = key
                    return flag, idx
                idx += 1

    def get_G0(self, chains, loops):
        assert self.is_loony_endgame(chains, loops)
        G0_chains, G0_loops = [], []
        for key in loops:
            if key.length >= 8:
                G0_loops.append(key)
        flag, num_3chain = 0, 0
        for key in chains:
            if key.length >= 4:
                flag = 1
                G0_chains.append(key)
            if key.length == 3:
                num_3chain += 1
        # assert num_3chain == 1 or num_3chain == 0
        if flag == 0 and num_3chain == 1:
            G0_chains.extend(chains)
        return G0_chains, G0_loops

    #   判断是不是loony_endgame
    def is_loony_endgame(self, chains, loops):  # 有无长度小于3的chain
        chains = sorted(chains, key=lambda point: point.length)
        loops = sorted(loops, key=lambda point: point.length)
        if len(chains) == 0 and len(loops) == 0:
            return True
        if len(chains) == 0:
            return True
        if chains[0].length < 3:
            return False
        return True

    def pd_unopen_game(self, board):  # edg_cnt只能为2和4
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

    def get_structure(self):  # 如果是opened那就分析去掉last_move的成分，如果是control，就分析把control填满的剩余部分的
        self.G_chains = []
        self.G_loops = []
        board_temp = np.array(self.board).reshape(self.args.board_size, self.args.board_size)
        if self.is_opened:
            board_temp[self.open_side[0]][self.open_side[1]] = 0
        elif self.is_control:
            for key in self.cnt3_side:
                board_temp, _ = self.game.get_next_state(board_temp, self.game.RED, key[0] * self.game.board_size + key[1])
        if not self.pd_unopen_game(board_temp):
            return False

        vis = [[0 for _ in range(self.args.board_size)] for _ in range(self.args.board_size)]
        for i in range(self.args.board_size):
            for j in range(self.args.board_size):
                if board_temp[i][j] == 0 and i % 2 == 1 and j % 2 == 1 and vis[i][j] == 0:
                    vis[i][j] = 1
                    move = self.get_unopen_move(board_temp, i, j, vis)
                    idx = []
                    for k in range(len(move)):
                        x, y = move[k]
                        if self.is_bound_side(x, y):
                            idx.append(k)
                    if len(idx) == 0:
                        loop = Loop(len(move), move)
                        self.G_loops.append(loop)
                    else:
                        move[:idx[0] + 1] = reversed(move[:idx[0] + 1])
                        chain = Chain(len(move) - 1, move)
                        # print((i, j), chain)
                        self.G_chains.append(chain)
        self.G_chains = sorted(self.G_chains, key=lambda point: point.length)
        self.G_loops = sorted(self.G_loops, key=lambda point: point.length)
        self.is_loony_game = self.is_loony_endgame(self.G_chains, self.G_loops)
        return True

    def get_unopen_move(self, board, x, y, vis):  # dfs
        move_list = []
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if board[nx][ny] == 0:
                move_list.append((nx, ny))
                board[nx][ny] = 1
            else:
                continue
            if 0 <= nx + dx[i] <= self.args.board_size-1 and self.args.board_size-1 >= ny + dy[i] >= 0 and vis[nx + dx[i]][ny + dy[i]] == 0:
                vis[nx + dx[i]][ny + dy[i]] = 1
                move_list.extend(self.get_unopen_move(board, nx + dx[i], ny + dy[i], vis))
        return move_list

    def is_end_game(self, board):
        board_temp = np.array(board).reshape(self.args.board_size, self.args.board_size)
        cnt0, cnt1, cnt2, cnt3 = 0, 0, 0, 0
        for i in range(self.args.board_size):
            for j in range(self.args.board_size):
                if i % 2 == 0 or j % 2 == 0:
                    continue
                edg_cnt = 0
                remain_x, remain_y = 0, 0
                for k in range(4):
                    nx = i + dx[k]
                    ny = j + dy[k]
                    if board_temp[nx][ny] != 0:
                        edg_cnt += 1
                    else:
                        remain_x, remain_y = nx, ny
                if edg_cnt == 0:
                    cnt0 += 1
                elif edg_cnt == 1:
                    cnt1 += 1
                elif edg_cnt == 2:
                    cnt2 += 1
                elif edg_cnt == 3:
                    cnt3 += 1
                    if (remain_x, remain_y) not in self.cnt3_side:
                        # print("box: ", (i, j), board_temp[i + nx[0]][j + ny[0]], board_temp[i + nx[1]][j + ny[1]],
                        #       board_temp[i + nx[2]][j + ny[2]], board_temp[i + nx[3]][j + ny[3]])
                        self.cnt3_side.append((remain_x, remain_y))
        if cnt0 == 0 and cnt1 == 0:
            self.is_endgame = True
        if self.is_endgame:
            if cnt0 == 0 and cnt1 == 0 and cnt3 == 0:
                self.un_open = True
            elif cnt3 == 1 or (cnt3 == 2 and len(self.cnt3_side) == 2):  # 后部分条件是open 2-chain 中间
                self.is_opened = True
            elif self.is_endgame and len(self.cnt3_side) == 1 and cnt3 == 2:  # control chain
                self.is_control = True
                self.control_chain = True
            elif self.is_endgame and cnt3 == 4:  # control loop
                self.is_control = True
                self.control_loop = True
        if self.is_opened or self.is_control:  # 需要分析链的结构，去掉last_move不能有01格子
            board_temp[self.open_side[0]][self.open_side[1]] = 0
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
                    if edg_cnt == 0 or edg_cnt == 1:
                        self.is_opened = False
                        self.is_control = False
                        self.is_endgame = False

    def is_bound_side(self, x, y):
        flag = False
        if x == 0 or x == self.args.board_size-1:
            flag = True
        if y == 0 or y == self.args.board_size-1:
            flag = True
        return flag

    def minimax_alpha_beta(self, alpha, beta, maximizing_player, player):  # alpha 有问题没修复
        if self.is_control and self.is_loony_game:
            # self.game.print_dots_and_boxes_board(self.board)
            # print(self.calculate_v(self.G_chains, self.G_loops))
            v = 0
            move = self.open(self.G_chains, self.G_loops)
            if self.control_chain:
                if maximizing_player:
                    v = 2 - self.calculate_v(self.G_chains, self.G_loops)
                else:
                    v = self.calculate_v(self.G_chains, self.G_loops) - 2
            elif self.control_loop:
                if maximizing_player:
                    v = 4 - self.calculate_v(self.G_chains, self.G_loops)
                else:
                    v = self.calculate_v(self.G_chains, self.G_loops) - 4
            temp = self.cnt3_side.copy()
            temp.append(move.move[0])
            return v, temp
        board_temp = np.array(self.board).reshape((11, 11))
        flag = True
        for i in range(11):
            for j in range(11):
                if i % 2 == 1 and j % 2 == 1:
                    if board_temp[i][j] == 0:
                        flag = False
        if flag:  # 游戏结束
            return 0, -1
        if maximizing_player:
            max_eval = -math.inf
            best_move = None
            move = self.get_possible_moves()
            board = self.get_board_by_move(move, player)
            for key in board:
                child_node = Endgame(key[0], key[3])
                child_node.get_structure()
                move = key[2]
                new_territory = key[1]
                eval, _ = child_node.minimax_alpha_beta(alpha - new_territory, beta, False, -player)
                if eval + new_territory > max_eval:
                    max_eval = eval + new_territory
                    best_move = move
                alpha = max(alpha, eval + new_territory)
                if beta <= alpha:
                    break  # Beta剪枝
            return max_eval, best_move
        else:
            min_eval = math.inf
            best_move = None
            move = self.get_possible_moves()
            board = self.get_board_by_move(move, player)
            for key in board:
                child_node = Endgame(key[0], key[3])
                child_node.get_structure()
                move = key[2]
                new_territory = key[1]
                eval, _ = child_node.minimax_alpha_beta(alpha, beta + new_territory, True, -player)
                if eval - new_territory < min_eval:
                    min_eval = eval - new_territory
                    best_move = move
                beta = min(beta, eval - new_territory)
                if beta <= alpha:
                    break  # Alpha剪枝
            return min_eval, best_move

    def abstract_minimax_alpha_beta(self, board, alpha, beta, maximizing_player, player, layer):
        flag = True
        if len(board) != 0:
            flag = False
        if flag:  # 游戏结束
            return 0, -1
        if maximizing_player:
            max_eval = -math.inf
            best_move = None
            board = self.abstract_get_child(board)
            for key in board:  # key = (board, v, move)
                eval, _ = self.abstract_minimax_alpha_beta(key[0], alpha - key[1], beta, False, -player, layer + 1)
                if eval + key[1] > max_eval:
                    max_eval = eval + key[1]
                    best_move = key[2]
                alpha = max(alpha, eval + key[1])
                if beta <= alpha:
                    break  # Beta剪枝
            return max_eval, best_move
        else:
            min_eval = math.inf
            best_move = None
            board = self.abstract_get_child(board)
            for key in board:
                eval, _ = self.abstract_minimax_alpha_beta(key[0], alpha, beta + key[1], True, -player, layer + 1)
                if eval - key[1] < min_eval:
                    min_eval = eval - key[1]
                    best_move = key[2]
                beta = min(beta, eval - key[1])
                if beta <= alpha:
                    break  # Alpha剪枝
            return min_eval, best_move
