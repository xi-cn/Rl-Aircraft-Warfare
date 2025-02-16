from keras import Sequential, Model
from keras.layers import Dense, TimeDistributed, LSTM, Conv2D, MaxPooling2D, Flatten, ConvLSTM2D
import keras.layers
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras import losses
import os
import cv2
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
from keras import Input
from keras.regularizers import l2

class DQN_LSTM(Model):

    def __init__(
        self, 
        n_actions, 
        gamma=0.9,
        learning_rate=0.001,
        *args, 
        **kwargs):
        super().__init__(*args, **kwargs)

        self.n_actions = n_actions
        self.gamma = gamma

        self.eval_net = self.build_net()
        self.target_net = self.build_net()

        self.eval_net.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=losses.MeanSquaredError()
        )
        self.optimizer = self.eval_net.optimizer
        self.lossf = self.eval_net.loss

        self.final_greedy = 0.99
        self.greedy_steps = 300000
        self.init_greedy = 0.01

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
    
    def call(self, inputs):
        return self.eval_net(inputs)
    
    def choose_action(self, s:np.ndarray):

        if self.step == 0:
            if np.random.uniform() > self.greedy:
                self.act_greedy = False
            else:
                self.act_greedy = True
            self.step = 10

        if not self.act_greedy:
            action = np.random.randint(0, self.n_actions)
            detail = np.zeros(shape=self.n_actions)
            detail[action] = 1
        else:
            output = self.eval_net(s)
            action = tf.argmax(output, axis=1).numpy()[0]
            detail = output.cpu().numpy()
            detail = np.squeeze(detail, axis=0)

        self.update_greedy()
        self.step -= 1
        return action, detail
    
    def learn(self, s, a, r, s_):

        with tf.GradientTape() as tape:
            q_next = self.target_net(s_)
            q_eval = self.eval_net(s)
            # 获取实际的q_target
            q_target = tf.identity(q_eval).cpu().numpy()

            batch_index = np.arange(q_next.shape[0], dtype=np.int32)
            q_next_max = tf.reduce_max(q_next, axis=1).cpu().numpy()
            # 用实际替代
            q_target[batch_index, a] = r + self.gamma * q_next_max
            q_target = tf.convert_to_tensor(q_target)
            # 计算损失值
            loss = self.lossf(q_eval, q_target)
        # 计算梯度
        gradients = tape.gradient(loss, self.eval_net.trainable_variables)
        # 更新权重
        self.optimizer.apply_gradients(zip(gradients, self.eval_net.trainable_variables))
    
    def covergeTargetNet(self):
        self.target_net.set_weights(self.eval_net.weights)
    

class Actor(Model):

    def __init__(
        self, 
        n_actions,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.n_actions = n_actions

        self.net = Sequential([
            TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), padding="same", activation='relu' )),
            TimeDistributed(MaxPooling2D(2, 2)),
            TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), padding="same", activation='relu' )),
            TimeDistributed(MaxPooling2D(2, 2)),
            TimeDistributed(Conv2D(64, (3, 3), strides=1, padding="same", activation='relu' )),
            TimeDistributed(Flatten()),
            LSTM(64),
            Dense(self.n_actions, activation='softmax' )
        ])
    
    def call(self, x):
        output = self.net(x)
        mx = tf.argmax(output, axis=1)
        one = tf.one_hot(mx, depth=self.n_actions)
        output = output * one
        return output
    
class Critic(Model):
    def __init__(
        self, 
        n_actions,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.n_actions = n_actions

        l_dim = 128
        self.feature_net  = Sequential([
            TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), padding="same", activation='relu')),
            TimeDistributed(MaxPooling2D(2, 2)),
            TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), padding="same", activation='relu')),
            TimeDistributed(MaxPooling2D(2, 2)),
            TimeDistributed(Conv2D(64, (3, 3), strides=1, padding="same", activation='relu')),
            TimeDistributed(Flatten()),
            LSTM(64),
        ])
        self.w_s = Dense(l_dim, use_bias=False)
        self.w_a = Dense(l_dim, use_bias=False)
        self.b = tf.Variable(tf.zeros(shape=(l_dim,)), trainable=True, name="b")
        self.final_net = Dense(1)


    def call(self, x, a):
        output = self.feature_net(x)
        output = tf.nn.relu(self.w_s(output) + self.w_a(a) + self.b)
        output = self.final_net(output)

        return output

import math

class DDPG(Model):
    def __init__(
        self, 
        n_actions=3,
        actor_lr=0.001,
        critic_lr=0.001,
        gamma=0.9,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.n_actions = n_actions
        self.gamma = gamma

        # 演员
        self.actor = Actor(n_actions=n_actions, trainable=True)
        self.actor_ = Actor(n_actions=n_actions, trainable=False)

        # 评论家
        self.critic = Critic(n_actions=n_actions, trainable=True)
        self.critic_ = Critic(n_actions=n_actions, trainable=False)

        # 编译模型
        self.actor.compile(
            optimizer=optimizers.Adam(learning_rate=actor_lr)
        )
        self.critic.compile(
            optimizer=optimizers.Adam(learning_rate=critic_lr)
        )

        # 优化器
        self.actor_optim = self.actor.optimizer
        self.critic_optim = self.critic.optimizer

        self.final_greedy = 0.9
        self.greedy_steps = 200000
        self.init_greedy = 0.01

        self.greedy_alpha=math.pow(self.final_greedy/self.init_greedy, 1/self.greedy_steps)
        self.greedy = self.init_greedy

    def build_model(self):
        s = tf.zeros(shape=(1, 11, 160, 160, 3))
        a = tf.zeros(shape=(1, 3))
        self.actor(s)
        self.actor_(s)
        self.critic(s, a)
        self.critic_(s, a)
        self.call(s)
        
    def call(self, s):
        return self.actor(s)

    # 更新贪婪指数
    def update_greedy(self):
        if self.greedy == self.final_greedy:
            return
        elif self.greedy > self.final_greedy:
            self.greedy = self.final_greedy
        else:
            self.greedy *= self.greedy_alpha

    # 选择行为
    def choose_action(self, s:np.ndarray):
        self.greedy = 1
        # 大于阈值 采取随机行为
        if np.random.uniform() > self.greedy:
            action = np.random.randint(0, self.n_actions)
            detail = np.zeros(shape=self.n_actions)
            detail[action] = 1
        else:
            output = self.actor(s)
            action = tf.argmax(output, axis=1).numpy()
            detail = output.cpu().numpy()
            detail = np.squeeze(detail, axis=0)

            res = self.critic(s, output)
            print(res)

        self.update_greedy()
        return action, detail
    
    # 覆盖演员网络
    def covergeActorNet(self):
        self.actor_.set_weights(self.actor.weights)
    # 覆盖评论家网络
    def covergeCriticNet(self):
        self.critic_.set_weights(self.critic.weights)

    # 学习函数
    def learn(self, s, a, r, s_):
        with tf.GradientTape() as tape:
            a_target = self.actor_(s_)
            q = self.critic(s, a)
            q_ = self.critic_(s_, a_target)
            # 目标
            target_q = r + self.gamma * q_
            # td error
            loss = losses.mean_squared_error(target_q, q)
        # 计算评论网络梯度
        critic_grad = tape.gradient(loss, self.critic.trainable_variables)
        # # 更新权重
        self.critic_optim.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        # 更新演员
        with tf.GradientTape() as tape:
            actor_loss = -self.critic(s, self.actor(s))
            actor_loss = tf.reduce_mean(actor_loss)
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optim.apply_gradients(zip(actor_grad, self.actor.trainable_variables))


class DataWithTest:
    def __init__(
        self,
        path="../data",
        exist_rate=0.2,
        interval=2,
        prev_num=10,
        sample_ration = (0.3, 0.4, 0.3),
        forth_step=5,
        batch_size=64,
        max_test_num=300000
    ):
        self.exist_rate=exist_rate
        self.interval=interval
        self.prev_num=prev_num
        self.forth_step=forth_step
        self.max_test_num=max_test_num
        self.batch_size=batch_size
        self.prev = prev_num * interval

        self.files = os.listdir(path)
        self.path = path + "/"

#         # 现有数据集索引映射
#         indices = []
#         for i, f in enumerate(self.files):
#             eps, step, action, reward = self.parseFileName(f)
#             if step < self.interval * self.prev_num or reward == 1000:
#                 continue
#             else:
#                 indices.append([i, reward])
#         self.exist_indices = np.array(indices)

#         # 对现有数据集进行分类
#         self.positive_exist = self.exist_indices[self.exist_indices[:,1] > 0, 0]
#         self.negtive_exist = self.exist_indices[self.exist_indices[:,1] < 0, 0]
#         self.zero_exist = self.exist_indices[self.exist_indices[:,1] == 0, 0]

        # # 直接读取图像到内存中
        # data = []
        # for i, f in enumerate(self.files):
        #     if i % 5000 == 0:
        #         print(f"已读取[{i}]张图像")
        #     img = cv2.imread(self.path + f).astype(np.float16)
        #     img = img/ 255.0
        #     data.append(img)
        # self.data = np.array(data)
        
#         print(f"图像读取完毕 总计[{len(self.data)}]张")
        
        # 测试数据信息
        self.test_images = []
        self.test_actions = []
        self.test_rewards = []
        self.test_positive = []
        self.test_zero = []

        # 负样本单独存储
        self.negtive_image = []
        self.negtive_reward = []
        self.negtive_action = []
        # 测试数据总数
        self.test_num = 0

        # 采样比例
        self.sample_ration = sample_ration
    
    # 新增数据集
    def addTestData(self, images, actions, rewards):
        self.test_images.append(images)
        self.test_actions.append(actions)
        self.test_rewards.append(rewards)
        # 样本分类
        test_positive = []
        test_zero = []
        for i in range(self.prev, len(rewards)):
            if rewards[i] > 0:
                test_positive.append(i)
            elif rewards[i] == 0:
                test_zero.append(i)
            elif rewards[i] < 0:
                low_index = i - self.interval * self.prev_num
                state = np.copy(images[low_index:i+1:self.interval])
                next_state = np.copy(images[low_index+1:i+2:self.interval])
                self.negtive_image.append((state, next_state))
                self.negtive_action.append(np.copy(actions[i]))
                self.negtive_reward.append(np.copy(rewards[i]))
                break

        self.test_positive.append(np.array(test_positive))
        self.test_zero.append(np.array(test_zero))
                
        
        # 更新测试数据总数
        self.test_num += len(images)
        # 维护测试数据总量
        while self.test_num > self.max_test_num:
            self.test_num -= len(self.test_rewards[0])
            # 移除旧的数据
            self.test_images.pop(0)
            self.test_actions.pop(0)
            self.test_rewards.pop(0)
            self.test_positive.pop(0)
            self.test_zero.pop(0)
        
        # 更新负数据
        while len(self.negtive_image) > 3000:
            self.negtive_action.pop(0)
            self.negtive_image.pop(0)
            self.negtive_reward.pop(0)
        

    
    # 根据文件名称解析得分
    def parseFileName(self, file):
        # 批次 步数 行为 得分
        eps, step, action, reward = [int(num) for num in file[0:-4].split("_")]
        if reward == -1 or reward == 1000:
            reward == -5

        return (eps, step, action, reward)

    # 从现有数据集中采样
    def chooseExistData(self, index):
        # 真正的索引
        real_index = index

        S = []
        R = []
        A = []
        S_ = []
        # 循环遍历索引
        for i in real_index:
            # 批次 步数 行为 得分
            (eps, step, action, reward) = self.parseFileName(self.files[i])
            R.append(reward)
            A.append(action)
            # 当前状态
            seq = range(i- self.prev, i+1, self.interval)
            S.append(self.data[seq])
            # 下一状态
            next_i = min(i+self.forth_step, len(self.data)-1)
            while True:
                (eps_, step_, action_, reward_) = self.parseFileName(self.files[next_i])
                if eps_ == eps:
                    break
                else:
                    next_i -= 1
            seq_ = range(next_i- self.prev, next_i+1, self.interval)
            S_.append(self.data[seq_])
        return np.array(S), np.array(R), np.array(A), np.array(S_)
    
    # 从测试数据中采样
    def chooseTestData(self, index):
        S = []
        A = []
        R = []
        S_ = []

        for (game_index, kind) in index:

            if not kind == -1:
                if kind == 0:
                    i = np.random.choice(self.test_zero[game_index], 1)[0]
                elif kind == 1 and len(self.test_positive[game_index]) > 0:
                    i = np.random.choice(self.test_positive[game_index], 1)[0]
                else:
                    i = np.random.choice(self.test_zero[game_index], 1)[0]

                low_index = i - self.interval * self.prev_num
                S.append(self.test_images[game_index][low_index:i+1:self.interval])
                A.append(self.test_actions[game_index][i])
                R.append(self.test_rewards[game_index][i])

                next_i = min(len(self.test_actions[game_index])-1, i+self.forth_step)
                low_index = next_i - self.interval * self.prev_num
                S_.append(self.test_images[game_index][low_index:next_i+1:self.interval])
            else:
                i = np.random.choice(len(self.negtive_action), 1)[0]
                S.append(self.negtive_image[i][0])
                S_.append(self.negtive_image[i][1])
                A.append(self.negtive_action[i])
                R.append(self.negtive_reward[i])

        return np.array(S), np.array(R), np.array(A), np.array(S_)


    def exist_sampling(self):
        negtive_num = int(self.batch_size * self.sample_ration[0])
        positive_num = int(self.batch_size * self.sample_ration[2])
        zero_num = self.batch_size - negtive_num - positive_num

        index1 = np.random.choice(self.negtive_exist, negtive_num)
        index2 = np.random.choice(self.zero_exist, zero_num)
        index3 = np.random.choice(self.positive_exist, positive_num)

        index = np.concatenate([index1, index2, index3], axis=-1)

        return index
    
    def test_sampling(self):
        index = np.random.choice(len(self.test_images), self.batch_size)
        kind = np.random.choice([-1, 0, 1], self.batch_size, p=self.sample_ration)

        result = np.stack([index, kind])
        result = result.transpose()

        return result

    # 获取训练数据
    def getTrainData(self):
        # 采用已有数据
        if np.random.uniform() < self.exist_rate:
            index = self.exist_sampling()
            return self.chooseExistData(index)
        # 采用测试数据
        else:
            index = self.test_sampling()
            return self.chooseTestData(index)

import keras

if __name__ == "__main__":
    x = tf.random.normal(shape=(10, 3))
    model = keras.layers.Softmax()

    x = model(x)
    mx = tf.argmax(x, axis=1)
    one = tf.one_hot(mx, depth=3)
    # one = tf.transpose(one)
    y = x * one
    print(y)
