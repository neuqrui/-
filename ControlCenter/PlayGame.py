import importlib
from TrainNet import TrainNet
from Storage import Log, Storage
import time
from xmlrpc.client import ServerProxy
import xmlrpc.client
from multiprocessing.managers import BaseManager
import queue
import progressbar
from threading import Thread
import gc

task_queue = queue.Queue()
result_queue = queue.Queue()
task_dict_queue = queue.Queue()
result_dict_queue = queue.Queue()
flag_computing = 0
flag_training = 0
data_list = []


class QueueManager(BaseManager):
    pass


class SelfPlay:
    """

    """

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
        self.mcts = self.Mcts(self.game, self.args, self.trainer, self.replay_buffer)
        self.server = None
        self.manager = None
        self.manager_dict = None
        self.num_train_data = self.args.num_train_rate * self.args.num_computing_center

        if args.load_latest_model:
            model_weight = self.memory.load_model_weight("best")
            self.trainer.net_work.load_state_dict(model_weight)
            self.memory.load_total_train_data(self.replay_buffer)
            self.num_iter, self.num_RED_win, self.num_BLUE_win, self.args.start_search_step_threshold, self.args.old_num_iter, old_avg_loss = self.memory.load_vs_info()
            # self.args.old_avg_loss[self.args.start_search_step_threshold] = old_avg_loss
            self.replay_buffer.merge_data(self.num_iter)
        else:
            self.memory.save_model_weight(self.trainer.net_work, "best")

    def run(self):
        """
        Generate board sequence through self-game
        :return:
        """
        self.server = ServerProxy(self.args.TrainCenterIP + ":" + self.args.TrainCenterPort)
        QueueManager.register('get_task_queue', callable=SelfPlay.re_task_queue)
        QueueManager.register('get_result_queue', callable=SelfPlay.re_result_queue)
        # Binding IP and Port, Set verification code:'abc'
        self.manager = QueueManager(address=(self.args.ControlCenterIP, self.args.ControlCenterPort), authkey=b'abc')
        self.manager.start()
        QueueManager.register('get_task_dict_queue', callable=SelfPlay.re_task_dict_queue)
        QueueManager.register('get_result_dict_queue', callable=SelfPlay.re_result_dict_queue)
        # Binding IP and Port, Set verification code:'abc'
        self.manager_dict = QueueManager(address=(self.args.ControlCenterIP, self.args.ControlCenterPort + 1),
                                         authkey=b'abc')
        self.manager_dict.start()

        while self.num_iter < self.args.num_iter:

            # Computing Thread
            if len(data_list) < 2 * self.num_train_data and flag_computing == 0:
                Log.print_string("\n*********************[ Start Computing Thread, data_list length: {}".format(
                    len(data_list)) + ' ]*********************\n')
                computing_thread = Thread(target=self.start_computing_task, args=(self.num_iter,))
                computing_thread.start()

            # Training Thread
            if len(data_list) >= self.num_train_data and flag_training == 0:
                Log.print_string("\n*********************[ Start Training Thread, data_list length: {}".format(
                    len(data_list)) + ' ]*********************\n')
                del self.replay_buffer
                gc.collect()
                root_Q_total_num, root_Q_right_num, search_step_root_Q_total_num, search_step_root_Q_right_num = 0, 0, 0, 0
                self.replay_buffer = self.replay_buffer_module.ReplayBuffer(self.args)
                for data_buffer, trajectory, num_RED_win, num_BLUE_win, a, b, c, d in data_list[:self.num_train_data]:
                    for num_iter, layers, visit_num, data in data_buffer:
                        self.replay_buffer.add_one_piece_data(num_iter, layers, visit_num, data)
                    for layers, board, net_pi, pi, net_v, v in trajectory:
                        self.replay_buffer.add_trajectory(layers, board, net_pi, pi, net_v, v)
                    self.num_RED_win += num_RED_win
                    self.num_BLUE_win += num_BLUE_win
                    root_Q_total_num += a
                    root_Q_right_num += b
                    search_step_root_Q_total_num += c
                    search_step_root_Q_right_num += d
                # 记录Q大于Q_threshold的比率
                self.memory.save_Q_threshold(num_iter, root_Q_total_num, root_Q_right_num,
                                             search_step_root_Q_total_num, search_step_root_Q_right_num,
                                             self.args.start_search_step_threshold)
                Log.print_vs_info(self.args, self.num_RED_win, self.num_BLUE_win)
                Log.print_str_int("Main Thread: Iter training data num", self.replay_buffer.get_iter_data_num())
                model_weight = self.memory.load_model_weight("best")
                self.trainer.net_work.load_state_dict(model_weight)
                step_loss_p, step_loss_v = self.trainer.calculate_step_loss(self.replay_buffer)
                self.move_forward(self.num_iter, step_loss_p, step_loss_v, root_Q_total_num, root_Q_right_num,
                                  search_step_root_Q_total_num, search_step_root_Q_right_num)
                self.memory.save_loss_step_p(step_loss_p)
                self.memory.save_loss_step_v(step_loss_v)
                # start = self.args.num_net * self.args.start_search_step_threshold
                # end = self.args.num_net * (
                #         self.args.start_search_step_threshold + self.args.interval_search_step_threshold)
                # interval = self.args.num_net * self.args.interval_search_step_threshold
                self.memory.save_vs_info(self.num_iter, self.num_RED_win, self.num_BLUE_win,
                                         self.args.start_search_step_threshold, self.args.old_num_iter)
                self.memory.save_iter_train_data(self.replay_buffer.iter_data)
                Log.print_string('Main Thread: Iter_Train_Data has been stored to disk!')
                data_list[:self.num_train_data] = []
                training_thread = Thread(target=self.start_train_task, args=(self.num_iter,))
                training_thread.start()
                self.num_iter += 1
            time.sleep(30)

    def start_computing_task(self, num_iter):
        global flag_computing
        global data_list
        flag_computing = 1
        """ 
        Multi-process operation to generate training data 
        """
        st_game = time.time()
        task = self.manager.get_task_queue()
        result = self.manager.get_result_queue()
        model = self.memory.load_model_data('model_best.pth.tar')
        Log.print_string('[Computing Thread]:  Assign tasks\n')
        time.sleep(3)
        p = progressbar.ProgressBar()
        p.start()   # 原先有self.args.num_computing_center
        for i in range(self.args.num_computing_center):
            p.update(i)
            task.put((model, self.args, num_iter))
        p.finish()
        Log.print_string('\n[Computing Thread]:  Wait to recover training data ......')
        # Wait for the worker to connect, calculate the result and return it
        for j in range(self.args.num_computing_center):
            try:
                if j == 0:
                    data_buffer, trajectory, num_white_win, num_black_win, root_Q_total_num, root_Q_right_num, search_step_root_Q_total_num, search_step_root_Q_right_num = result.get(block=True, timeout=None)
                    data_list.append([data_buffer, trajectory, num_white_win, num_black_win, root_Q_total_num, root_Q_right_num, search_step_root_Q_total_num, search_step_root_Q_right_num])
                    Log.print_string("[Computing Thread]:  Got num " + str(j + 1) + " training data!  Time: " + str(
                        time.strftime('%H:%M:%S', time.localtime(time.time()))))
                else:
                    data_buffer, trajectory, num_white_win, num_black_win, root_Q_total_num, root_Q_right_num, search_step_root_Q_total_num, search_step_root_Q_right_num = result.get(block=True,
                                                                                       timeout=self.args.timeout_recycle)
                    data_list.append([data_buffer, trajectory, num_white_win, num_black_win, root_Q_total_num, root_Q_right_num, search_step_root_Q_total_num, search_step_root_Q_right_num])
                    Log.print_string("[Computing Thread]:  Got num " + str(j + 1) + " training data!  Time: " + str(
                        time.strftime('%H:%M:%S', time.localtime(time.time()))))
            except Exception as e:
                print('[Computing Thread]: Timeout stop recovering training data: {}'.format(e))
        et_game = time.time()
        Log.print_string(
            '[Computing Thread]: Iter: ' + str(num_iter) + ' Got all train data from Computing Center, cost: ' + str(
                "%.2f" % (et_game - st_game)) + ' s!')
        self.memory.save_play_time(num_iter, (et_game - st_game))
        flag_computing = 0

    def start_train_task(self, num_iter):
        global flag_training
        flag_training = 1
        st_time = time.time()
        data = open("./data/" + self.args.GAME_NAME + "/iter_train_data.pth.tar.examples", 'rb')
        self.upload_train_data("./data/" + self.args.GAME_NAME + "/", data, "iter_train_data", ".pth.tar.examples")
        data.close()
        Log.print_string('[Training Thread]: Iter training data has been sent to server, cost: ' + str(
            "%.2f" % (time.time() - st_time)) + ' s!')
        st_time = time.time()

        self.server.train(num_iter)
        Log.print_string('[Training Thread]: The training data has been trained， cost: ' + str(
            "%.2f" % (time.time() - st_time)) + ' s!')
        
        self.memory.save_train_time(num_iter, (time.time() - st_time))
        st_time = time.time()
        self.download_model(num_iter)
        Log.print_string('[Training Thread]: The latest model has been received, cost: ' + str(
            "%.2f" % (time.time() - st_time)) + 's!')
        flag_training = 0

    def upload_train_data(self, folder, data, file_name, format_name):
        try:
            self.server.download_iter_train_data(folder, xmlrpc.client.Binary(data.read()), file_name, format_name)
        except Exception as e:
            Log.print_string(e)
            self.upload_train_data(folder, data, file_name, format_name)

    def download_model(self, num_iter):
        model_name = './data/' + self.args.GAME_NAME + "/" + 'model_best.pth.tar'
        get_handle = open(model_name, 'wb')
        try:
            get_handle.write(self.server.upload_model(model_name).data)
        except Exception as e:
            Log.print_string(e)
            get_handle.write(self.server.upload_model(model_name).data)
        get_handle.close()
        self.memory.save_model_weight(self.trainer.net_work, str(num_iter))

    def move_forward(self, num_iter, loss_p, loss_v, root_Q_total_num, root_Q_right_num, search_step_root_Q_total_num,
                     search_step_root_Q_right_num):
        # start = self.args.num_net * self.args.start_search_step_threshold
        # end = self.args.num_net * (self.args.start_search_step_threshold + self.args.interval_search_step_threshold)
        # interval = self.args.num_net * self.args.interval_search_step_threshold
        start = self.args.num_net * self.args.start_search_step_threshold
        end = self.args.num_net * self.args.num_max_layers
        interval = end - start
        avg_loss = sum(loss_p[start:]) / interval
        Log.print_string('avg_loss: ' + str(round(avg_loss, 4)))
        Q_rate = round(root_Q_right_num / root_Q_total_num, 4)
        search_step_Q_rate = round(search_step_root_Q_right_num / search_step_root_Q_total_num, 4)
        print("Q_rate: ", Q_rate)
        if (num_iter - self.args.old_num_iter) >= self.args.delta_num_iter and self.args.start_search_step_threshold > 0:
            # if num_iter - self.args.old_num_iter >= 100 and self.args.start_search_step_threshold > 15:
            #     self.args.start_search_step_threshold -= self.args.delta_search_step_threshold
            #     print("满足deta_iter >= 100，前进一步！！！！！！")
            #     return None
            if self.args.update_iter_by_Q and (
                    not Q_rate >= self.args.root_Q_threshold_rate or not search_step_Q_rate >= self.args.start_search_step_root_Q_threshold_rate):
                print("不满足Q约束，不前进！！！！！！")
                return None
            self.args.old_num_iter = num_iter
            if self.args.start_search_step_threshold >= self.args.delta_search_step_threshold:
                self.args.start_search_step_threshold -= self.args.delta_search_step_threshold
                print("满足约束，前进一步！！！！！！")
            else:
                self.args.start_search_step_threshold = 0

    @staticmethod
    def re_task_queue():
        global task_queue
        return task_queue

    @staticmethod
    def re_result_queue():
        global result_queue
        return result_queue

    @staticmethod
    def re_task_dict_queue():
        global task_dict_queue
        return task_dict_queue

    @staticmethod
    def re_result_dict_queue():
        global result_dict_queue
        return result_dict_queue
