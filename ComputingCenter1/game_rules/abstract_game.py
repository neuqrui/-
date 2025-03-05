from abc import ABC, abstractmethod


class AbstractArgs(ABC):
    @abstractmethod
    def __init__(self):
        pass


class AbstractGame(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def init_board(self):
        return None

    @abstractmethod
    def play_one_game(self, in_layer: int, in_board: list, in_player: int, proc_id: int, num_iter: int, mcts: object, policy: str):
        pass

    @abstractmethod
    def get_next_state(self, board, player, action):
        return None

    @abstractmethod
    def get_legal_action(self, board, layers, start_pos=None):
        return None

    @abstractmethod
    def game_over(self, board, player=None):
        return None

    @abstractmethod
    def change_perspectives(self, board, player):
        return None

    @abstractmethod
    def to_string(self, board):
        return None
