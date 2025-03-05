import importlib
from xmlrpc.server import SimpleXMLRPCServer
from socketserver import ThreadingMixIn
from TrainNet import TrainNet
from FileTransfer import FileTransfer
from Storage import Log, Storage
import time


class ThreadXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass

# from 分布式.ControlCenter.game_args.dotsboxes_6 import A
class SelfPlay:
    def __init__(self, game, Mcts, args):
        """
        :param Mcts:(class)
        :param game:(object)
        :param args:(object)
        """
        self.game = game
        self.args = args
        self.num_RED_win = 0
        self.num_BLUE_win = 0
        self.num_iter = 1
        str_list = args.GAME_NAME.split('_')
        self.replay_buffer_module = importlib.import_module("replay_buffer." + str_list[0])
        self.replay_buffer = self.replay_buffer_module.ReplayBuffer(self.args)
        self.memory = Storage(self.game, self.args)
        self.model_module = importlib.import_module("model." + args.GAME_NAME)
        self.trainer = TrainNet(self.args, self.model_module.NetWork(self.args), self.memory)
        self.args.num_params = sum(param.numel() for param in self.trainer.net_work.parameters())
        Log.print_args(self.args)
        self.Mcts = Mcts
        self.mcts = None
        if self.args.load_latest_model:
            model_weight = self.memory.load_model_weight("best")
            self.trainer.net_work.load_state_dict(model_weight)
            self.memory.load_total_train_data(self.replay_buffer)
            self.num_iter, self.num_RED_win, self.num_BLUE_win, self.args.start_search_step_threshold, self.args.old_num_iter, _ = self.memory.load_vs_info()
            self.num_iter = int(self.num_iter) + 1
        else:
            self.memory.save_model_weight(self.trainer.net_work, "best")

    def run(self):
        """
        Generate board sequence through self-game
        :return:
        """
        server = ThreadXMLRPCServer((self.args.TrainCenterIP, self.args.TrainCenterPort), allow_none=True)
        server.register_function(self.train, 'train')
        server.register_function(FileTransfer.upload_model, 'upload_model')
        server.register_function(FileTransfer.download_iter_train_data, 'download_iter_train_data')
        print('Listening for Client......')
        server.serve_forever()

    def train(self, num_iter):
        Log.print_iter_num(num_iter)
        if num_iter != 1:
            self.memory.load_total_train_data(self.replay_buffer)
        self.memory.load_iter_train_data(self.replay_buffer)
        Log.print_str_int("Iter training data num", self.replay_buffer.get_iter_data_num())
        print("-----------self.replay_buffer.merge_data(num_iter)------------")
        self.replay_buffer.merge_data(num_iter)
        self.memory.save_step_data_num(num_iter, self.replay_buffer.get_step_data_num())
        Log.print_str_int("Total training data num", self.replay_buffer.get_total_data_num())
        st_game = time.time()
        self.trainer.train(self.replay_buffer, num_iter)
        self.memory.save_time_train(num_iter, (time.time() - st_game))
        self.memory.save_model_weight(self.trainer.net_work, str(num_iter))
        self.memory.save_model_weight(self.trainer.net_work, "best")
        self.memory.save_total_train_data(self.replay_buffer.total_data)
