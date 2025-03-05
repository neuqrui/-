from abc import ABC, abstractmethod


class AbstractMcts(ABC):

    @abstractmethod
    def __init__(self, game, args, trainer, replay_buffer):
        pass

    @abstractmethod
    def search(self, num_iter, board, layers, action=None):
        pass

    @abstractmethod
    def select_action(self, proc, board, num_iter, layers, policy):
        return None

    @abstractmethod
    def extract_data(self, node, num_iter):
        pass

    @abstractmethod
    def get_action_by_policy(self, counts, policy):
        return None


class AbstractNode(ABC):

    @abstractmethod
    def __init__(self, game, board, move, layers, parent=None, flag=False):
        pass

    @property
    @abstractmethod
    def my_n(self):
        pass

    @my_n.setter
    @abstractmethod
    def my_n(self, num):
        return None

    @property
    @abstractmethod
    def my_q(self):
        pass

    @my_q.setter
    @abstractmethod
    def my_q(self, value):
        return None

    @abstractmethod
    def select_child_by_uct(self, num_iter):
        pass

    @abstractmethod
    def expand_child(self, num_iter):
        return None

    @abstractmethod
    def expand_myself(self, child_pi):
        pass

    @abstractmethod
    def backup(self, value):
        pass
