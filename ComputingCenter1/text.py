def train(self, replay_buffer, num_iter):
    """
    examples: list of examples, each example is of form (board, pi, v)
    :param replay_buffer: (object) Experience RePlay Pool Object
    :param num_iter: (int): num of iteration
    """
    self.net_work.train())
    optimizer = optim.Adam(self.net_work.parameters(), lr=lr, weight_decay=self.args.weight_decay)
    data_num = replay_buffer.get_total_data_num()

    for epoch in range(self.args.epochs):
        print('EPOCH ::: ' + str(epoch + 1))
        batch_idx = 0
        while batch_idx < int(data_num / self.args.batch_size):
            data, target_policy, target_values, iter_num, num_lay = replay_buffer.get_batch()
            out_pi, out_v = self.net_work(data)
            l_reg = self.reg_loss(self.net_work)
            iter_np = torch.tensor(np.array([num_iter for _ in range(iter_num[i].shape[0])]))
            iter_deca = torch.pow(self.args.replay_decay_rate, (iter_np.contiguous().cuda() - iter_num[i]))
            l_pi += self.loss_pi(target_policy[i], out_pi, iter_deca)
            l_v += self.loss_v(target_values[i], out_v, iter_deca)
            total_loss = l_pi + l_v + l_reg
            # compute gradient and do Adam step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_idx += 1
