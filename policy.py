import tensorflow as tf
from network import *
from environment import *
import time
import copy

# node policy for train
class NodePolicy:

    def __init__(self,sub, n_actions, n_features,num_epoch, batch_size, learning_rate=0.01):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.num_epoch = num_epoch
        self.batch_size = batch_size

        self.ep_obs, self.ep_as, self.ep_rs,self.req_as = [], [], [], []
        self.sub=copy.deepcopy(sub)
        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_actions, self.n_features, 1],
                                         name="observations")
            self.tf_acts = tf.placeholder(tf.int32,  name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="action_value")


        with tf.name_scope('conv'):
            kernel=tf.Variable(tf.truncated_normal([1,self.n_features,1,1],dtype=tf.float32,stddev=0.1),
                               name="weights")
            conv = tf.nn.conv2d(input=self.tf_obs,
                                filter=kernel,
                                strides=(1, 1,self.n_features,1),
                                padding='VALID')
            biases=tf.Variable(tf.constant(0.0,shape=[1],dtype=tf.float32),name="bias")
            conv1=tf.nn.relu(tf.nn.bias_add(conv,biases))
            self.scores=tf.reshape(conv1,[-1,self.n_actions])

        with tf.name_scope('output'):
            self.probs=tf.nn.softmax(self.scores)

        # loss function
        with tf.name_scope('loss'):
            # 获取策略网络中全部可训练的参数
            self.tvars = tf.trainable_variables()
            # 设置虚拟label的placeholder
            self.input_y = tf.placeholder(tf.float32, [None, self.n_actions], name="input_y")
            # 计算损失函数loss(当前Action对应的概率的对数)
            self.loglik = -tf.reduce_sum(tf.log(self.probs) * self.input_y, axis=1)
            self.loss = tf.reduce_mean(self.loglik)
            # 计算损失函数梯度
            self.newGrads = tf.gradients(self.loss, self.tvars)


        # Optimizer
        with tf.name_scope('update'):
            # 权重参数梯度
            self.kernel_grad = tf.placeholder(tf.float32, name="batch_grad1")
            # 偏置参数梯度
            self.biases_grad = tf.placeholder(tf.float32, name="batch_grad2")
            # 整合两个梯度
            self.batch_grad = [self.kernel_grad, self.biases_grad]
            # 优化器
            adam = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.update_grads = adam.apply_gradients(zip(self.batch_grad, self.tvars))

    def choose_action(self, observation, sub, current_node_cpu,curreqnum):
        x = np.reshape(observation, [1, observation.shape[0], observation.shape[1], 1])
        prob_weights = self.sess.run(self.scores,
                                     feed_dict={self.tf_obs: x})

        candidate_action = []
        candidate_score = []
        for index, score in enumerate(prob_weights.ravel()):
            if index not in self.req_as and sub.nodes[index]['cpu_remain'] >= current_node_cpu:
                candidate_action.append(index)
                candidate_score.append(score)
        if len(candidate_action) == 0:
            return -1
        else:
            candidate_prob = np.exp(candidate_score) / np.sum(np.exp(candidate_score))
            action = np.random.choice(candidate_action, p=candidate_prob)

        self.req_as.append(action)
        if len(self.req_as) == curreqnum:
            self.req_as=[]

        return action

    # def choose_max_action(self,observation,sub,current_node_cpu,curreqnum):
    #     x = np.reshape(observation, [1, observation.shape[0], observation.shape[1], 1])
    #     prob_weights = self.sess.run(self.scores,
    #                                  feed_dict={self.tf_obs: x})
    #
    #     candidate_action = []
    #     candidate_score = []
    #     for index, score in enumerate(prob_weights.ravel()):
    #         if index not in self.req_as and sub.nodes[index]['cpu_remain'] >= current_node_cpu:
    #             candidate_action.append(index)
    #             candidate_score.append(score)
    #     if len(candidate_action) == 0:
    #         return -1
    #     else:
    #         candidate_prob = np.exp(candidate_score) / np.sum(np.exp(candidate_score))
    #         action_index = candidate_prob.index(np.max(candidate_prob))
    #         action = candidate_action[action_index]
    #
    #     self.req_as.append(action)
    #     if len(self.req_as) == curreqnum:
    #         self.req_as = []
    #
    #     return action

    def train(self, training_set):

        loss_average = []
        iteration = 0
        start = time.time()
        # 训练开始
        while iteration < self.num_epoch:
            values = []
            print("Iteration %s" % iteration)
            # 每轮训练开始前，都需要重置底层网络和相关的强化学习环境
            sub_copy = copy.deepcopy(self.sub)
            env = NodeEnv(self.sub)
            # 创建存储参数梯度的缓冲器
            grad_buffer = self.sess.run(self.tvars)
            # 初始化为0
            for ix, grad in enumerate(grad_buffer):
                grad_buffer[ix] = grad * 0
            # 记录已经处理的虚拟网络请求数量
            counter = 0
            # 获得底层网络的状态
            obsreset = env.reset()
            mapped_info = {}
            xs, acts = [], []
            reqr, reqc = 0, 0
            for req in training_set:

                if req.graph['type'] == 0:
                    print('req%d is mapping... ' % req.graph['id'])
                    print('node mapping...')
                    counter += 1
                    # 向环境传入当前的待映射虚拟网络
                    env.set_vnr(req)
                    node_map = {}
                    for vn_id in range(req.number_of_nodes()):
                        observation = obsreset
                        x = np.reshape(observation, [1, observation.shape[0], observation.shape[1], 1])
                        sn_id = self.choose_action(observation, env.sub, req.nodes[vn_id]['cpu'], req.number_of_nodes())
                        if sn_id == -1:
                            break
                        else:
                            # 输入的环境信息添加到xs列表中
                            xs.append(x)
                            # 将选择的动作添加到acts列表中
                            acts.append(sn_id)
                            # 执行一次action，获取返回的四个数据
                            observation_next, _, done, info = env.step(sn_id)
                            obsreset = observation_next
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
                        else:
                            obsreset = env.statechange(node_map)
                            print('req%d mapping is failed ' % req.graph['id'])
                    else:
                        obsreset = env.statechange(node_map)
                        print('req%d mapping is failed ' % req.graph['id'])

                    if counter % 10 == 0:
                        if reqc != 0:
                            reward = reqr / reqc
                        else:
                            reward = 0

                        if reward != 0:
                            ys = tf.one_hot(acts, self.n_actions)
                            epx = np.vstack(xs)
                            epy = tf.Session().run(ys)

                            # 返回损失函数值
                            loss_value = self.sess.run(self.loss,
                                                       feed_dict={self.tf_obs: epx,
                                                                  self.input_y: epy})
                            print('%d episode is done and the loss is' % (counter / 10), loss_value)
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

                        xs, acts = [], []
                        reqr, reqc = 0, 0

                    # 当实验次数达到batch size整倍数，累积的梯度更新一次参数
                    if counter % self.batch_size == 0:
                        print("update grads")
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
        with open('Results/losslog-%s.txt' % self.num_epoch, 'w') as f:
            f.write("Training time: %s hours\n" % end)
            for value in loss_average:
                f.write(str(value))
                f.write('\n')

    def run(self):
        pass


# link policy for train
class LinkPolicy:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.95,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features[0], self.n_features[1], 1],
                                         name="observations")
            self.tf_acts = tf.placeholder(tf.int32,  name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="action_value")

        with tf.name_scope('conv'):
            kernel=tf.Variable(tf.truncated_normal([1,self.n_features[1],1,1],dtype=tf.float32,stddev=0.1),
                               name="weights")
            conv = tf.nn.conv2d(input=self.tf_obs,
                                filter=kernel,
                                strides=(1, 1,self.n_features[1],1),
                                padding='VALID')
            biases=tf.Variable(tf.constant(0.0,shape=[1],dtype=tf.float32),name="bias")
            conv1=tf.nn.relu(tf.nn.bias_add(conv,biases))
            self.scores=tf.reshape(conv1,[-1,self.n_features[0]])

        with tf.name_scope('output'):
            self.probs=tf.nn.softmax(self.scores)

        # loss function
        with tf.name_scope('loss'):
            self.tvars=tf.trainable_variables()
            self.neg_log_prob = tf.reduce_sum(-tf.log(self.probs)*tf.one_hot(self.tf_acts,self.n_actions),axis=1)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.tf_vt)
            self.newGrads=tf.gradients(self.loss,self.tvars)


        # Optimizer
        with tf.name_scope('train'):
            # 权重参数梯度
            self.kernel_grad = tf.placeholder(tf.float32, name="batch_grad1")
            # 偏置参数梯度
            self.biases_grad = tf.placeholder(tf.float32, name="batch_grad2")
            # 整合两个梯度
            self.batch_grad = [self.kernel_grad, self.biases_grad]
            # 优化器
            adam = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = adam.apply_gradients(zip(self.batch_grad,self.tvars))

    def choose_action(self, observation,sub, linkbw, linkpath,vfr,vto):
        x = np.reshape(observation, [1, observation.shape[0], observation.shape[1], 1])
        prob_weights = self.sess.run(self.scores,
                                     feed_dict={self.tf_obs: x})

        candidate_action = []
        candidate_score = []
        for index, score in enumerate(prob_weights.ravel()):
            s_fr = list(linkpath[index].keys())[0][0]
            s_to = list(linkpath[index].keys())[0][1]
            v_fr = vfr
            v_to = vto
            if s_fr==v_fr and s_to==v_to and minbw(sub,list(linkpath[index].values())[0]) >= linkbw:
                candidate_action.append(index)
                candidate_score.append(score)
        if len(candidate_action) == 0:
            return -1
        else:
            candidate_prob = np.exp(candidate_score) / np.sum(np.exp(candidate_score))
            action = np.random.choice(candidate_action, p=candidate_prob)

        return action

    # def choose_max_action(self,observation, sub, vnr, linkpath,vfr,vto):
    #     x = np.reshape(observation, [1, observation.shape[0], observation.shape[1], 1])
    #     prob_weights = self.sess.run(self.scores,
    #                                  feed_dict={self.tf_obs: x})
    #
    #     candidate_action = []
    #     candidate_score = []
    #     for index, score in enumerate(prob_weights.ravel()):
    #         s_fr = list(linkpath[index].keys())[0][0]
    #         s_to = list(linkpath[index].keys())[0][1]
    #         v_fr = vfr
    #         v_to = vto
    #         if s_fr == v_fr and s_to == v_to and minbw(sub,linkpath[index]) >= vnr[v_fr][v_to]['bw']:
    #             candidate_action.append(index)
    #             candidate_score.append(score)
    #     if len(candidate_action) == 0:
    #         return -1
    #     else:
    #         candidate_prob = np.exp(candidate_score) / np.sum(np.exp(candidate_score))
    #         action_index = candidate_prob.index(np.max(candidate_prob))
    #         action = candidate_action[action_index]
    #
    #     return action

    def store_transition(self, s, a, r):
        s = np.reshape(s, [s.shape[0], s.shape[1], 1])
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)



    # discount episode rewards
    def discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # discounted_ep_rs -= np.mean(discounted_ep_rs)
        # discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    #train model

    def learn(self,lrcre,recnum):
        discounted_ep_rs_norm = self.discount_and_norm_rewards()
        discounted_ep_rs_norm = [i*lrcre*recnum for i in discounted_ep_rs_norm]
        discounted_ep_rs_norm -= np.mean(discounted_ep_rs_norm)
        discounted_ep_rs_norm /= np.std(discounted_ep_rs_norm)
        discounted_ep_rs_norm = [i*10000 for i in discounted_ep_rs_norm]


        # 返回求解梯度
        tf_grad = self.sess.run(self.newGrads,
                                feed_dict={self.tf_obs: self.ep_obs,
                                           self.tf_acts: self.ep_as,
                                           self.tf_vt: discounted_ep_rs_norm})
        # 创建存储参数梯度的缓冲器
        grad_buffer = self.sess.run(self.tvars)
        # 将获得的梯度累加到gradBuffer中
        for ix, grad in enumerate(tf_grad):
            grad_buffer[ix] += grad

        self.sess.run(self.train_op,
                      feed_dict={self.kernel_grad: grad_buffer[0],
                                 self.biases_grad: grad_buffer[1]})

        y=self.sess.run(self.loss, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs).reshape(len(self.ep_obs), self.n_features[0], self.n_features[1], 1),
            self.tf_acts: np.array(self.ep_as),
            self.tf_vt: discounted_ep_rs_norm,
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []


        return y


# node policy for test by use trained policy
class nodepolicy:
    def __init__(self,
                 n_actions,
                 n_features,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self._build_net()
        self.req_as = []
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features[0], self.n_features[1], 1],
                                         name="observations")

        with tf.name_scope('conv'):
            read=tf.train.NewCheckpointReader('./nodemodel/nodemodel.ckpt')

            kernel=read.get_tensor('conv/weights')
            conv = tf.nn.conv2d(input=self.tf_obs,
                                filter=kernel,
                                strides=(1, 1,self.n_features[1],1),
                                padding='VALID')
            biases=read.get_tensor('conv/bias')
            conv1=tf.nn.relu(tf.nn.bias_add(conv,biases))
            self.scores=tf.reshape(conv1,[-1,self.n_features[0]])

        with tf.name_scope('output'):
            self.probs=tf.nn.softmax(self.scores)

    def choose_max_action(self,observation,sub,current_node_cpu,curreqnum):
        x = np.reshape(observation, [1, observation.shape[0], observation.shape[1], 1])
        prob_weights = self.sess.run(self.scores,
                                     feed_dict={self.tf_obs: x})

        candidate_action = []
        candidate_score = []
        for index, score in enumerate(prob_weights.ravel()):
            if index not in self.req_as and sub.nodes[index]['cpu_remain'] >= current_node_cpu:
                candidate_action.append(index)
                candidate_score.append(score)
        if len(candidate_action) == 0:
            return -1
        else:
            action_index = candidate_score.index(np.max(candidate_score))
            action = candidate_action[action_index]

        self.req_as.append(action)
        if len(self.req_as) == curreqnum:
            self.req_as = []

        return action



# link policy for test by use trained policy

class linkpolicy:
    def __init__(self,
                 n_actions,
                 n_features):
        self.n_actions = n_actions
        self.n_features = n_features
        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features[0], self.n_features[1], 1],
                                         name="observations")
            self.tf_acts = tf.placeholder(tf.int32,  name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="action_value")

        with tf.name_scope('conv'):

            read2=tf.train.NewCheckpointReader("./linkmodel/linkmodel.ckpt")
            kernel=read2.get_tensor('conv/weights')
            conv = tf.nn.conv2d(input=self.tf_obs,
                                filter=kernel,
                                strides=(1, 1,self.n_features[1],1),
                                padding='VALID')
            biases=read2.get_tensor('conv/bias')
            conv1=tf.nn.relu(tf.nn.bias_add(conv,biases))
            self.scores=tf.reshape(conv1,[-1,self.n_features[0]])

        with tf.name_scope('output'):
            self.probs=tf.nn.softmax(self.scores)

    def choose_max_action(self,observation,sub, bw, linkpath,vfr,vto):
        x = np.reshape(observation, [1, observation.shape[0], observation.shape[1], 1])
        prob_weights = self.sess.run(self.scores,
                                     feed_dict={self.tf_obs: x})

        candidate_action = []
        candidate_score = []
        for index, score in enumerate(prob_weights.ravel()):
            s_fr = list(linkpath[index].keys())[0][0]
            s_to = list(linkpath[index].keys())[0][1]
            v_fr = vfr
            v_to = vto
            if s_fr == v_fr and s_to == v_to and minbw(sub,list(linkpath[index].values())[0]) >= bw:
                candidate_action.append(index)
                candidate_score.append(score)
        if len(candidate_action) == 0:
            return -1
        else:
            action_index = candidate_score.index(np.max(candidate_score))
            action = candidate_action[action_index]

        return action


# class NodePolicy:
#
#     def __init__(self,
#                  n_actions,
#                  n_features,
#                  learning_rate=0.01,
#                  reward_decay=0.95,
#                  ):
#         self.n_actions = n_actions
#         self.n_features = n_features
#         self.lr = learning_rate
#         self.gamma = reward_decay
#
#         self.ep_obs, self.ep_as, self.ep_rs,self.req_as = [], [], [], []
#         self._build_net()
#         self.sess = tf.Session()
#         self.sess.run(tf.global_variables_initializer())
#
#
#     def _build_net(self):
#         with tf.name_scope('inputs'):
#             self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features[0], self.n_features[1], 1],
#                                          name="observations")
#             self.tf_acts = tf.placeholder(tf.int32,  name="actions_num")
#             self.tf_vt = tf.placeholder(tf.float32, [None, ], name="action_value")
#
#
#         with tf.name_scope('conv'):
#             kernel=tf.Variable(tf.truncated_normal([1,self.n_features[1],1,1],dtype=tf.float32,stddev=0.1),
#                                name="weights")
#             conv = tf.nn.conv2d(input=self.tf_obs,
#                                 filter=kernel,
#                                 strides=(1, 1,self.n_features[1],1),
#                                 padding='VALID')
#             biases=tf.Variable(tf.constant(0.0,shape=[1],dtype=tf.float32),name="bias")
#             conv1=tf.nn.relu(tf.nn.bias_add(conv,biases))
#             self.scores=tf.reshape(conv1,[-1,self.n_features[0]])
#
#         with tf.name_scope('output'):
#             self.probs=tf.nn.softmax(self.scores)
#
#         # loss function
#         with tf.name_scope('loss'):
#             self.tvars=tf.trainable_variables()
#             self.neg_log_prob = tf.reduce_sum(-tf.log(self.probs)*tf.one_hot(self.tf_acts,self.n_actions),axis=1)
#             self.loss = tf.reduce_mean(self.neg_log_prob * self.tf_vt)
#             self.newGrads=tf.gradients(self.loss,self.tvars)
#
#
#         # Optimizer
#         with tf.name_scope('train'):
#             # 权重参数梯度
#             self.kernel_grad = tf.placeholder(tf.float32, name="batch_grad1")
#             # 偏置参数梯度
#             self.biases_grad = tf.placeholder(tf.float32, name="batch_grad2")
#             # 整合两个梯度
#             self.batch_grad = [self.kernel_grad, self.biases_grad]
#             # 优化器
#             adam = tf.train.AdamOptimizer(learning_rate=self.lr)
#             self.train_op = adam.apply_gradients(zip(self.batch_grad,self.tvars))
#
#     def choose_action(self, observation, sub, current_node_cpu,curreqnum):
#         x = np.reshape(observation, [1, observation.shape[0], observation.shape[1], 1])
#         prob_weights = self.sess.run(self.scores,
#                                      feed_dict={self.tf_obs: x})
#
#         candidate_action = []
#         candidate_score = []
#         for index, score in enumerate(prob_weights.ravel()):
#             if index not in self.req_as and sub.nodes[index]['cpu_remain'] >= current_node_cpu:
#                 candidate_action.append(index)
#                 candidate_score.append(score)
#         if len(candidate_action) == 0:
#             return -1
#         else:
#             candidate_prob = np.exp(candidate_score) / np.sum(np.exp(candidate_score))
#             action = np.random.choice(candidate_action, p=candidate_prob)
#
#         self.req_as.append(action)
#         if len(self.req_as) == curreqnum:
#             self.req_as=[]
#
#         return action
#
#     # def choose_max_action(self,observation,sub,current_node_cpu,curreqnum):
#     #     x = np.reshape(observation, [1, observation.shape[0], observation.shape[1], 1])
#     #     prob_weights = self.sess.run(self.scores,
#     #                                  feed_dict={self.tf_obs: x})
#     #
#     #     candidate_action = []
#     #     candidate_score = []
#     #     for index, score in enumerate(prob_weights.ravel()):
#     #         if index not in self.req_as and sub.nodes[index]['cpu_remain'] >= current_node_cpu:
#     #             candidate_action.append(index)
#     #             candidate_score.append(score)
#     #     if len(candidate_action) == 0:
#     #         return -1
#     #     else:
#     #         candidate_prob = np.exp(candidate_score) / np.sum(np.exp(candidate_score))
#     #         action_index = candidate_prob.index(np.max(candidate_prob))
#     #         action = candidate_action[action_index]
#     #
#     #     self.req_as.append(action)
#     #     if len(self.req_as) == curreqnum:
#     #         self.req_as = []
#     #
#     #     return action
#
#     def store_transition(self, s, a, r):
#         s = np.reshape(s, [s.shape[0], s.shape[1], 1])
#         self.ep_obs.append(s)
#         self.ep_as.append(a)
#         self.ep_rs.append(r)
#
#
#     # discount episode rewards
#     def discount_and_norm_rewards(self):
#         discounted_ep_rs = np.zeros_like(self.ep_rs)
#         running_add = 0
#         for t in reversed(range(0, len(self.ep_rs))):
#             running_add = running_add * self.gamma + self.ep_rs[t]
#             discounted_ep_rs[t] = running_add
#
#         # discounted_ep_rs -= np.mean(discounted_ep_rs)
#         # discounted_ep_rs /= np.std(discounted_ep_rs)
#         return discounted_ep_rs
#
#     #train model
#
#     def learn(self,lrcre,recnum):
#         discounted_ep_rs_norm = self.discount_and_norm_rewards()
#         discounted_ep_rs_norm = [i*lrcre*recnum for i in discounted_ep_rs_norm]
#         discounted_ep_rs_norm -= np.mean(discounted_ep_rs_norm)
#         discounted_ep_rs_norm /= np.std(discounted_ep_rs_norm)
#         discounted_ep_rs_norm = [i * 10000 for i in discounted_ep_rs_norm]
#
#         # 返回求解梯度
#         tf_grad = self.sess.run(self.newGrads,
#                                 feed_dict={self.tf_obs: self.ep_obs,
#                                            self.tf_acts: self.ep_as,
#                                            self.tf_vt: discounted_ep_rs_norm})
#         # 创建存储参数梯度的缓冲器
#         grad_buffer = self.sess.run(self.tvars)
#         # 将获得的梯度累加到gradBuffer中
#         for ix, grad in enumerate(tf_grad):
#             grad_buffer[ix] += grad
#
#         self.sess.run(self.train_op,
#                       feed_dict={self.kernel_grad: grad_buffer[0],
#                                  self.biases_grad: grad_buffer[1]})
#
#         y=self.sess.run(self.loss, feed_dict={
#             self.tf_obs: np.vstack(self.ep_obs).reshape(len(self.ep_obs), self.n_features[0], self.n_features[1], 1),
#             self.tf_acts: np.array(self.ep_as),
#             self.tf_vt: discounted_ep_rs_norm,
#         })
#
#         self.ep_obs, self.ep_as, self.ep_rs = [], [], []
#
#         return y