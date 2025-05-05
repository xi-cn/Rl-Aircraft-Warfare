from keras import Sequential, Model
from keras.layers import Dense, TimeDistributed, LSTM, Conv2D, MaxPooling2D, Flatten
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras import losses
from keras import Input
import math

class DQN(Model):
    def __init__(
        self, 
        n_actions, 
        gamma=0.9,
        learning_rate=0.001,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # 动作空间
        self.n_actions = n_actions
        self.gamma = gamma

        # 构建网络
        self.eval_net = self.build_net()
        self.target_net = self.build_net()
        # 评估网络
        self.eval_net.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=losses.MeanSquaredError()
        )
        self.optimizer = self.eval_net.optimizer
        self.lossf = self.eval_net.loss

        # greedy策略
        self.final_greedy = 0.99
        self.greedy_steps = 10000
        self.init_greedy = 0.1
        self.greedy_alpha=math.pow(self.final_greedy/self.init_greedy, 1/self.greedy_steps)
        self.greedy = self.init_greedy

        self.step = 0
        self.act_greedy = False
    
    # 更新贪婪指数
    def update_greedy(self):
        if self.greedy == self.final_greedy:
            return
        elif self.greedy > self.final_greedy:
            self.greedy = self.final_greedy
        else:
            self.greedy *= self.greedy_alpha

    # 构建网络
    def build_net(self):
        net = Sequential([
            TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), padding="same", activation='relu')),
            TimeDistributed(MaxPooling2D(2, 2)),
            TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), padding="same", activation='relu')),
            TimeDistributed(MaxPooling2D(2, 2)),
            TimeDistributed(Conv2D(64, (3, 3), strides=1, padding="same", activation='relu')),
            TimeDistributed(Flatten()),
            LSTM(512),
            Dense(self.n_actions)
        ])

        return net
    
    def build(self, input_shape):
        super().build(input_shape)
        self.target_net.build(input_shape)
        self.eval_net.build(input_shape)
    
    def call(self, inputs):
        return self.eval_net(inputs)
    
    # 选择行为
    def choose_action(self, s:np.ndarray):
        # 当选择greedy策略后 连续执行10次
        if self.step == 0:
            if np.random.uniform() > self.greedy:
                self.act_greedy = False
            else:
                self.act_greedy = True
            self.step = 10
        # 随机选择动作
        if not self.act_greedy:
            action = np.random.randint(0, self.n_actions)
            detail = np.zeros(shape=self.n_actions)
            detail[action] = 1
        # 模型选择动作
        else:
            output = self.eval_net(s)
            action = tf.argmax(output, axis=1).numpy()[0]
            detail = output.cpu().numpy()
            detail = np.squeeze(detail, axis=0)

        self.update_greedy()
        self.step -= 1
        return action, detail
    
    # 更新函数
    def learn(self, s, a, r, s_):

        with tf.GradientTape() as tape:
            q_next = self.target_net(s_)
            q_eval = self.eval_net(s)
            # 获取实际的q_target
            q_target = tf.identity(q_eval).cpu().numpy()

            batch_index = np.arange(q_next.shape[0], dtype=np.int32)
            q_next_max = tf.reduce_max(q_next, axis=1).cpu().numpy()

            # # 游戏结束的样本
            # done_sample = r == -5
            # q_target[batch_index[done_sample], a[done_sample]] = r[done_sample]
            # # 未结束的样本
            # q_target[batch_index[~done_sample], a[~done_sample]] = r[~done_sample] + self.gamma * q_next_max[~done_sample]
            q_target[batch_index, a] = r + self.gamma * q_next_max

            q_target = tf.convert_to_tensor(q_target)
            # 计算损失值
            loss = self.lossf(q_eval, q_target)
        # 计算梯度
        gradients = tape.gradient(loss, self.eval_net.trainable_variables)
        # 更新权重
        self.optimizer.apply_gradients(zip(gradients, self.eval_net.trainable_variables))
        return loss.cpu().numpy()
    
    # 覆盖网络
    def covergeTargetNet(self):
        self.target_net.set_weights(self.eval_net.weights)


# double DQN
class DDQN(DQN):
    # 构造函数
    def __init__(
        self, 
        n_actions, 
        gamma=0.9,
        learning_rate=0.001,
        *args, 
        **kwargs
    ):
        super().__init__(n_actions, gamma, learning_rate, *args, **kwargs)
    
    # 重载更新函数
    def learn(self, s, a, r, s_):
        with tf.GradientTape() as tape:
            # Q(St+1, target_net)
            q_next = tf.stop_gradient(self.target_net(s_)).cpu().numpy()
            # Q(St, eval_net)
            q_eval = self.eval_net(s)
            # Q(St+1, eval_net)
            q_eval_next = tf.stop_gradient(self.eval_net(s_)).cpu().numpy()

            # q_eval的复制体
            q_target = tf.identity(q_eval).cpu().numpy()

            # 要更新的索引
            batch_index = np.arange(q_eval_next.shape[0], dtype=np.int32)
            q_next_index = np.argmax(q_eval_next, axis=1)

            # 目标值
            q_next_target = q_next[batch_index, q_next_index]

            # # 游戏结束的样本
            # done_sample = r == -5
            # q_target[batch_index[done_sample], a[done_sample]] = r[done_sample]
            # # 未结束的样本
            # q_target[batch_index[~done_sample], a[~done_sample]] = r[~done_sample] + self.gamma * q_next_target[~done_sample]
            q_target[batch_index, a] = r + self.gamma * q_next_target

            q_target = tf.convert_to_tensor(q_target)
            # 计算损失值
            loss = self.lossf(q_eval, q_target)

        # 计算梯度
        gradients = tape.gradient(loss, self.eval_net.trainable_variables)
        # 更新权重
        self.optimizer.apply_gradients(zip(gradients, self.eval_net.trainable_variables))
        return loss.cpu().numpy()


# Dueling DQN
class DuelingDQN(DQN):
    # 构造函数
    def __init__(
        self, 
        n_actions, 
        gamma=0.9,
        learning_rate=0.001,
        *args, 
        **kwargs
    ):
        super().__init__(n_actions, gamma, learning_rate, *args, **kwargs)

    # 构建网络
    def build_net(self):

        inputs = Input(shape=(None, 160, 160, 3))

        output = Sequential([
            TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), padding="same", activation='relu')),
            TimeDistributed(MaxPooling2D(2, 2)),
            TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), padding="same", activation='relu')),
            TimeDistributed(MaxPooling2D(2, 2)),
            TimeDistributed(Conv2D(64, (3, 3), strides=1, padding="same", activation='relu')),
            TimeDistributed(Flatten()),
            LSTM(512),
        ])(inputs)
        # state value
        v_out = Dense(1)(output)
        v_out = tf.tile(v_out, multiples=[1, self.n_actions])
        # advanteage
        a_out = Dense(self.n_actions)(output)
        # advantage average
        a_mean = tf.reduce_mean(a_out, axis=1)
        a_mean = tf.expand_dims(a_mean, axis=1)
        a_mean = tf.tile(a_mean, multiples=[1, self.n_actions])
        a_out = a_out - a_mean

        outputs = v_out + a_out

        model = Model(inputs=inputs, outputs=outputs)
        return model

# Raingbow DQN
class RainbowDQN(DQN):
    # 构造函数
    def __init__(
        self, 
        n_actions, 
        gamma=0.9,
        learning_rate=0.001,
        *args, 
        **kwargs
    ):
        super().__init__(n_actions, gamma, learning_rate, *args, **kwargs)

if __name__ == "__main__":
    import numpy as np
    X = np.random.ranf(size=(8, 11, 160, 160, 3))

    model = DuelingDQN(5)
    y = model.target_net(X)
    print(y)
    print(y.shape)
