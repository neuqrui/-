import importlib
from PlayGame import SelfPlay
from Storage import Logger, Log
import sys


class DeltaZero:
    """
    The main interface of the running program

    Example:
        >>> delta_zero = DeltaZero('amazons_10')
        >>> delta_zero.self_learn()
    """
    def __init__(self, game_args):
        """
        Load the config and the Specified module with the game_rules name

        :param game_args:(str) the game_rules name
        :return:None
        """

        # Determine the specified game_rules module according to the parameters

        game_args_module = importlib.import_module("game_args." + game_args)
        self.args = game_args_module.Args()
        str_list = game_args.split('_')
        game_rules_module = importlib.import_module("game_rules." + str_list[0])
        self.Game = game_rules_module.Game
        mcts_module = importlib.import_module("mcts." + str_list[0])
        self.Mcts = mcts_module.Mcts
        sys.stdout = Logger(self.args.GAME_NAME, "log_main_process.log", sys.stdout)
        # sys.stderr = Logger(self.game_args.GAME_NAME, "error.log", sys.stderr)
        self.game = None
        self.self_play = None
        self.trainer = None
        self.memory = None

    def self_learn(self):
        """
        First generate data from the game_rules, and then iteratively train the network
        :return:None
        """
        Log.print_string("Start self-playing and training network!")
        Log.print_string(f"Args logs: board_size:[{self.args.board_size}*{self.args.board_size}]; load_latest_model:{'[Yes]' if self.args.load_latest_model else '[No]'}")
        self.game = self.Game(self.args)
        self.self_play = SelfPlay(self.game, self.Mcts, self.args)
        self.self_play.run()
