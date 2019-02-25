import tensorflow as tf
from network import *

# node policy for train
class NodePolicy:

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

        self.ep_obs, self.ep_as, self.ep_rs,self.req_as = [], [], [], []
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
        discounted_ep_rs_norm = [i * 10000 for i in discounted_ep_rs_norm]

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
            # kernel=tf.constant([[[[-1.1386735]],[[0.27748054]],[[0.4875062]],[[0.43316445]],[[-0.01042795]]]],dtype=tf.float32)
            conv = tf.nn.conv2d(input=self.tf_obs,
                                filter=kernel,
                                strides=(1, 1,self.n_features[1],1),
                                padding='VALID')
            biases=read.get_tensor('conv/bias')
            # biases=tf.constant([0.6251562])
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


