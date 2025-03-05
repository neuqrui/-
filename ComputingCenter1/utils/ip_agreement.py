import re
import time
import logging
import datetime
import numpy as np


class Agreement:
    def __init__(self, game, skt):
        self.command_dict = {'@dz_who': self.who,
                             '@dz_iam': self.iam,
                             '@dz_info': self.info,
                             '@dz_move': self.move,
                             '@dz_wait': self.wait,
                             '@dz_quit': self.quit,
                             '@dz_board': self.board,
                             '@dz_pk_num': self.pk_num,
                             '@dz_request_boards': self.request_boards,
                             '@dz_game_over': self.game_over
                             }
        self.game = game
        self.skt = skt

    def get_info(self, msg=None):
        key, msgs = self.skt.recv(1024).decode().split(':')
        return self.command_dict[key](msgs)

    def send_info(self, msg=None):
        self.skt.send(msg.encode())

    def who(self, msg=None):
        return msg

    def iam(self, msg=None):
        return msg

    def info(self, msg=None):
        pass

    def board(self, msg=None):
        """
        把字符串转换成棋盘矩阵
        :param msg:
        :return: board: np.array
        """
        return np.array([int(x) for x in re.findall(r"-?\d+\.*\d*", msg)], dtype=np.int32).reshape((self.game.board_size, self.game.board_size))

    def move(self, msg=None):

        pass

    def wait(self, msg=None):
        pass

    def quit(self, msg=None):
        pass

    def pk_num(self, msg=None):
        return int(msg) if msg is not None else None

    def request_boards(self, msg=None):
        return msg

    def game_over(self, msg=None):
        pass


class Log:
    @staticmethod
    def write_log(msg, level):
        print(msg)
        log_level = {'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR}
        logger = logging.getLogger()
        filename = time.strftime('%Y-%m-%d', time.localtime(time.time()))

        handler = logging.FileHandler("./log/" + filename + ".log")
        logger.addHandler(handler)
        logger.setLevel(log_level[level])
        logger.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': ' + msg)
        logger.removeHandler(handler)
