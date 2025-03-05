import torch
import math
from game_rules.abstract_game import AbstractArgs


class Args(AbstractArgs):
    """
    Parameters class
    """

    def __init__(self):
        super().__init__()
        # 这个board_size是指能表示 6 * 6 的点格棋的二维矩阵的大小
        self.board_size = 11
        # 这个point_size = 6 指的是点的个数
        self.point_num = 6
        # 一共有60条边
        self.side_size = 2 * self.point_num * (self.point_num - 1)
        # 一共有25块领地
        self.site_size = (self.point_num - 1) ** 2
        # 多少条边，max_layers的值就是多少
        self.num_max_layers = self.side_size
        self.GAME_NAME = 'dotsboxes_6'
        # Mcts search
        self.num_iter = 1000
        self.num_play_game = 10     # The number of games played in one iteration
        self.train_num_search = 2000  # simulation search times
        self.train_num_search_before = 0
        self.sharpening_policy_t = 0.4  # t power processing of children N 0.4
        self.Cpuct = 1
        # Delta train
        self.start_search_step_threshold = 32  # 真实的搜索博弈起点 50
        # self.interval_search_step_threshold = 3   # 选择数据的真实的步数间隔
        self.delta_search_step_threshold = 2  # 每次移动的步数
        self.delta_num_iter = 30  # 每次前进时必须满足的迭代次数间隔
        self.old_num_iter = 0  # 记录上一次更新时的迭代次数

        # replay_buffer
        self.N_threshold = self.train_num_search * 0.15
        self.N_Q_threshold1 = 0.80
        self.N_Q_threshold2 = 0.70
        self.N_Q_threshold3 = 0.85
        self.N_Q_threshold4 = 0.95
        self.N_threshold_before = 5
        self.N_Q_threshold_before = 0
        self.replay_decay_rate = 0.98
        self.replay_buffer_threshold = 8    # replay_buffer 中只留最近5次迭代的数据
        self.min_data_num = 5000000  # one iteration 至少向replay_buffer中放入棋盘的数量
        self.min_game_num = 5  # one iteration 至少play game的数量
        # NetWork
        self.load_latest_model = False
        self.num_net = 1
        self.lr = 0.0003
        self.weight_decay = 0.00001
        self.lr_iter_threshold = 80  # 大于80-iteration学习率减半
        self.epochs = 5
        self.batch_size = 512
        self.num_params = 0
        self.C_pi_loss = 1
        self.C_v_loss = math.sqrt(self.board_size)
        self.C_reg_loss = 0

        # Process
        self.multiprocess = False  # Multi-process switch
        self.num_process = 4  # Number of multiple processes
        self.SingleProcess = -1  # Single process id
        self.print_log_process = 0  # Print the output of the process id in the terminal
        # Gpu Parallel
        self.cuda = torch.cuda.is_available()
        self.gpu_parallel = False
        self.gpu_num = torch.cuda.device_count()
        self.gpu_ids = range(0, torch.cuda.device_count(), 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.EPS = 1e-10
        # old and new networks PK
        self.Pk = True
        self.update_iter_threshold = 0.5
        self.num_pk = 2
        self.start_pk_iter_threshold = 20
        # Compared with other AI engines
        self.first_hand = True  # First hand uses white chess and works as a server
        self.interface_pk_step_threshold = 5
        self.IP_interface_pk = '192.168.1.106'
        self.Port_interface_pk = 10000
        # 新增改进
        self.use_mct_constraint = True                      # mct中的约束策略
        self.input_channels = 26                            # 网络输入为2个/26通道
        self.update_iter_by_Q = True                        # 根据q值决定是否前进一步， 控制Q值有两层分别对应N
        self.root_Q_threshold = 0.85                        # 根节点的network——>Q绝对值约束
        self.root_Q_threshold_rate = 0.75                   # 满足Q约束的比例大于这个rate才可以前进
        self.start_search_step_root_Q_threshold = 0.85      # 树根真实Q约束
        self.start_search_step_root_Q_threshold_rate = 0.85
        self.start_search_step_root_Q_threshold_interval = 4
        self.train_num_search_front_step = 3000
        self.train_num_search_front_step_interval = 2     # 2000 search step_num = 4 + 1
        # Distributed Computing
        self.TrainCenterIP = "127.0.0.1"
        self.TrainCenterPort = 20001 # 61112