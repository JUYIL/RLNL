#rlvne (A novel reinforcement learning ...)

#agent
import tensorflow as tf
import numpy as np
import copy
import time
from network import bfslinkmap



class RL:

    def __init__(self, sub, n_actions, n_features, learning_rate, num_epoch, batch_size):
        self.n_actions = n_actions  # 动作空间大小
        self.n_features = n_features  # 节点向量维度
        self.lr = learning_rate  # 学习速率
        self.num_epoch = num_epoch  # 训练轮数
        self.batch_size = batch_size  # 批处理的批次大小

        self.sub = copy.deepcopy(sub)
        self._build_model()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self, training_set):

        loss_average = []
        iteration = 0
        start = time.time()
        # 训练开始
        while iteration < self.num_epoch:
            values = []
            print("Iteration %s" % iteration)
            # 每轮训练开始前，都需要重置底层网络和相关的强化学习环境
            env = Env(self.sub)
            # 创建存储参数梯度的缓冲器
            grad_buffer = self.sess.run(self.tvars)
            # 初始化为0
            for ix, grad in enumerate(grad_buffer):
                grad_buffer[ix] = grad * 0
            # 记录已经处理的虚拟网络请求数量
            counter = 0
            mapped_info = {}
            obsreset = env.reset()
            for req in training_set:

                if req.graph['type'] == 0:
                    reqr, reqc = 0, 0
                    xs, acts = [], []
                    print('req%d is mapping... ' % req.graph['id'])
                    print('node mapping...')
                    counter += 1
                    # 向环境传入当前的待映射虚拟网络
                    env.set_vnr(req)
                    node_map = {}
                    for vn_id in range(req.number_of_nodes()):
                        observation = obsreset
                        x = np.reshape(observation, [1, observation.shape[0], observation.shape[1], 1])
                        sn_id = self.choose_action(observation, self.sub, req.nodes[vn_id]['cpu'], acts)
                        if sn_id == -1:
                            break
                        else:
                            # 输入的环境信息添加到xs列表中
                            xs.append(x)
                            # 将选择的动作添加到acts列表中
                            acts.append(sn_id)
                            # 执行一次action，获取返回的四个数据
                            obsreset, _, done, info = env.step(sn_id)
                            node_map.update({vn_id: sn_id})
                    # end for,即一个VNR的全部节点映射全部尝试完毕

                    if len(node_map) == req.number_of_nodes():
                        print('link mapping...')
                        link_map = bfslinkmap(env.sub, req, node_map)
                        if len(link_map) == req.number_of_edges():
                            print('req%d is mapped ' % req.graph['id'])
                            for node in req.nodes:
                                reqr += req.nodes[node]['cpu']
                            reqc = reqr
                            for vl, pl in link_map.items():
                                vfr, vto = vl[0], vl[1]
                                reqr += req[vfr][vto]['bw']
                                reqc += req[vfr][vto]['bw'] * (len(pl) - 1)
                                i = 0
                                while i < len(pl) - 1:
                                    env.sub[pl[i]][pl[i + 1]]['bw_remain'] -= req[vfr][vto]['bw']
                                    i += 1
                            mapped_info.update({req.graph['id']: (node_map, link_map)})
                            reward=reqr/reqc
                            ys = tf.one_hot(acts, self.n_actions)
                            epx = np.vstack(xs)
                            epy = tf.Session().run(ys)

                            # 返回损失函数值
                            loss_value = self.sess.run(self.loss,
                                                       feed_dict={self.tf_obs: epx,
                                                                  self.input_y: epy})
                            print("Success! The loss value is: %s" % loss_value)
                            values.append(loss_value)

                            # 返回求解梯度
                            tf_grad = self.sess.run(self.newGrads,
                                                    feed_dict={self.tf_obs: epx,
                                                               self.input_y: epy})
                            # 将获得的梯度累加到gradBuffer中
                            for ix, grad in enumerate(tf_grad):
                                grad_buffer[ix] += grad
                            grad_buffer[0] *= reward
                            grad_buffer[1] *= reward
                        else:
                            obsreset = env.statechange(node_map)
                            print('req%d mapping is failed ' % req.graph['id'])
                    else:
                        obsreset = env.statechange(node_map)
                        print('req%d mapping is failed ' % req.graph['id'])

                    # 当实验次数达到batch size整倍数，累积的梯度更新一次参数
                    if counter % self.batch_size == 0:
                        self.sess.run(self.update_grads,
                                      feed_dict={self.kernel_grad: grad_buffer[0],
                                                 self.biases_grad: grad_buffer[1]})

                        # 清空gradBuffer
                        for ix, grad in enumerate(grad_buffer):
                            grad_buffer[ix] = grad * 0

                if req.graph['type'] == 1:
                    if mapped_info.__contains__(req.graph['id']):
                        print('req%d is leaving... ' % req.graph['id'])
                        env.set_vnr(req)
                        reqid = req.graph['id']
                        nodemap = mapped_info[reqid][0]
                        linkmap = mapped_info[reqid][1]
                        for vl, path in linkmap.items():
                            i = 0
                            while i < len(path) - 1:
                                env.sub[path[i]][path[i + 1]]['bw_remain'] += req[vl[0]][vl[1]]['bw']
                                i += 1
                        obsreset = env.statechange(nodemap)
                        mapped_info.pop(reqid)
                    else:
                        pass

            loss_average.append(np.mean(values))
            iteration = iteration + 1

        end = (time.time() - start) / 3600
        with open('Results/c1losslog-%s.txt' % self.num_epoch, 'w') as f:
            f.write("Training time: %s hours\n" % end)
            for value in loss_average:
                f.write(str(value))
                f.write('\n')


    def _build_model(self):
        """搭建策略网络"""

        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(dtype=tf.float32,
                                         shape=[None, self.n_actions, self.n_features, 1],
                                         name="observations")

            self.tf_acts = tf.placeholder(dtype=tf.int32,
                                          shape=[None, ],
                                          name="actions_num")

            self.tf_vt = tf.placeholder(dtype=tf.float32,
                                        shape=[None, ],
                                        name="action_value")

        with tf.name_scope("conv"):
            self.kernel = tf.Variable(tf.truncated_normal([1, self.n_features, 1, 1],
                                                          dtype=tf.float32,
                                                          stddev=0.1),
                                      name="weights")
            conv = tf.nn.conv2d(input=self.tf_obs,
                                filter=self.kernel,
                                strides=[1, 1, self.n_features, 1],
                                padding="VALID")
            self.bias = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                                    name="bias")
            conv1 = tf.nn.relu(tf.nn.bias_add(conv, self.bias))
            self.scores = tf.reshape(conv1, [-1, self.n_actions])

        with tf.name_scope("output"):
            self.probability = tf.nn.softmax(self.scores)

        # 损失函数
        with tf.name_scope('loss'):
            # 获取策略网络中全部可训练的参数
            self.tvars = tf.trainable_variables()
            # 设置虚拟label的placeholder
            self.input_y = tf.placeholder(tf.float32, [None, self.n_actions], name="input_y")
            # 计算损失函数loss(当前Action对应的概率的对数)
            self.loglik = -tf.reduce_sum(tf.log(self.probability) * self.input_y, axis=1)
            self.loss = tf.reduce_mean(self.loglik)
            # 计算损失函数梯度
            self.newGrads = tf.gradients(self.loss, self.tvars)

        # 批量梯度更新
        with tf.name_scope('update'):
            # 权重参数梯度
            self.kernel_grad = tf.placeholder(tf.float32, name="batch_grad1")
            # 偏置参数梯度
            self.biases_grad = tf.placeholder(tf.float32, name="batch_grad2")
            # 整合两个梯度
            self.batch_grad = [self.kernel_grad, self.biases_grad]
            # 优化器
            adam = tf.train.AdamOptimizer(learning_rate=self.lr)
            # 累计到一定样本梯度，执行updateGrads更新参数
            self.update_grads = adam.apply_gradients(zip(self.batch_grad, self.tvars))

    def choose_action(self, observation, sub, current_node_cpu, acts):
        """在给定状态observation下，根据策略网络输出的概率分布选择动作，供训练阶段使用，兼顾了探索和利用"""

        # 规范化网络输入格式
        x = np.reshape(observation, [1, observation.shape[0], observation.shape[1], 1])

        tf_score = self.sess.run(self.scores, feed_dict={self.tf_obs: x})
        candidate_action = []
        candidate_score = []
        for index, score in enumerate(tf_score.ravel()):
            if index not in acts and sub.nodes[index]['cpu_remain'] >= current_node_cpu:
                candidate_action.append(index)
                candidate_score.append(score)

        if len(candidate_action) == 0:
            return -1
        else:
            candidate_prob = np.exp(candidate_score) / np.sum(np.exp(candidate_score))
            # 选择动作
            action = np.random.choice(candidate_action, p=candidate_prob)
            return action



#Env
import gym
from gym import spaces
import networkx as nx
from network import calculate_adjacent_bw


class Env(gym.Env):

    def render(self, mode='human'):
        pass

    def __init__(self, sub):
        self.count = -1
        self.n_action = sub.number_of_nodes()
        self.sub = copy.deepcopy(sub)
        self.action_space = spaces.Discrete(self.n_action)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_action, 4), dtype=np.float32)
        self.state = None
        self.actions = []
        self.degree = []
        for i in nx.degree_centrality(sub).values():
            self.degree.append(i)
        self.vnr = None

    def set_vnr(self, vnr):
        self.vnr = vnr
        self.count=-1

    def step(self, action):
        self.actions.append(action)
        self.count = self.count + 1
        cpu_remain, bw_all_remain, avg_dst = [], [], []
        for u in range(self.n_action):
            adjacent_bw = calculate_adjacent_bw(self.sub, u, 'bw_remain')
            if u == action:
                self.sub.nodes[action]['cpu_remain'] -= self.vnr.nodes[self.count]['cpu']
                adjacent_bw -= calculate_adjacent_bw(self.vnr, self.count)
            cpu_remain.append(self.sub.nodes[u]['cpu_remain'])
            bw_all_remain.append(adjacent_bw)

            sum_dst = 0
            for v in self.actions:
                sum_dst += nx.shortest_path_length(self.sub, source=u, target=v)
            sum_dst /= (len(self.actions) + 1)
            avg_dst.append(sum_dst)

        cpu_remain = (cpu_remain - np.min(cpu_remain)) / (np.max(cpu_remain) - np.min(cpu_remain))
        bw_all_remain = (bw_all_remain - np.min(bw_all_remain)) / (np.max(bw_all_remain) - np.min(bw_all_remain))
        avg_dst = (avg_dst - np.min(avg_dst)) / (np.max(avg_dst)-np.min(avg_dst))

        self.state = (cpu_remain,
                      bw_all_remain,
                      self.degree,
                      avg_dst)
        return np.vstack(self.state).transpose(), 0.0, False, {}

    def statechange(self, nodemap):
        self.count = -1
        self.actions = []
        cpu_remain, bw_all_remain = [], []

        for vid, sid in nodemap.items():
            self.sub.nodes[sid]['cpu_remain'] += self.vnr.nodes[vid]['cpu']
        for u in range(self.n_action):
            cpu_remain.append(self.sub.nodes[u]['cpu_remain'])
            bw_all_remain.append(calculate_adjacent_bw(self.sub, u, 'bw_remain'))
        for vid, sid in nodemap.items():
            bw_all_remain[sid]+=calculate_adjacent_bw(self.vnr, vid)

        cpu_remain = (cpu_remain - np.min(cpu_remain)) / (np.max(cpu_remain) - np.min(cpu_remain))
        bw_all_remain = (bw_all_remain - np.min(bw_all_remain)) / (np.max(bw_all_remain) - np.min(bw_all_remain))
        avg_dst = np.zeros(self.n_action).tolist()

        self.state = (cpu_remain,
                      bw_all_remain,
                      self.degree,
                      avg_dst)
        return np.vstack(self.state).transpose()

    def reset(self):
        """获得底层网络当前最新的状态"""
        self.count = -1
        self.actions = []
        cpu_remain, bw_all_remain = [], []
        for u in range(self.n_action):
            cpu_remain.append(self.sub.nodes[u]['cpu_remain'])
            bw_all_remain.append(calculate_adjacent_bw(self.sub, u, 'bw_remain'))

        cpu_remain = (cpu_remain - np.min(cpu_remain)) / (np.max(cpu_remain) - np.min(cpu_remain))
        bw_all_remain = (bw_all_remain - np.min(bw_all_remain)) / (np.max(bw_all_remain) - np.min(bw_all_remain))
        avg_dst = np.zeros(self.n_action).tolist()
        self.state = (cpu_remain,
                      bw_all_remain,
                      self.degree,
                      avg_dst)
        return np.vstack(self.state).transpose()