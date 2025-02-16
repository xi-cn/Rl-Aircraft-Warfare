from keras import Sequential, Model
from keras.layers import Dense, TimeDistributed, LSTM, Conv2D, MaxPooling2D, Flatten
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras import losses
import os
import cv2
import math
import sys

class DQN(Model):

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
            Conv2D(32, (8, 8), strides=(4, 4), padding="same", activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, (4, 4), strides=(2, 2), padding="same", activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), strides=1, padding="same", activation='relu'),
            Flatten(),
            Dense(512, activation='relu'),
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


class DataWithTest:
    def __init__(
        self,
        sample_ration = (0.3, 0.4, 0.3),
        forth_step=5,
        batch_size=64,
        max_test_num=300000
    ):
        self.forth_step=forth_step
        self.max_test_num=max_test_num
        self.batch_size=batch_size

        
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
        for i in range(len(rewards)):
            if rewards[i] > 0:
                test_positive.append(i)
            elif rewards[i] == 0:
                test_zero.append(i)
            elif rewards[i] < 0:
                state = np.copy(images[i])
                next_state = np.copy(images[i+1])
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
            a1 = self.test_images.pop(0)
            a2 = self.test_actions.pop(0)
            a3 = self.test_rewards.pop(0)
            a4 = self.test_positive.pop(0)
            a5 = self.test_zero.pop(0)
            del a1, a2, a3, a4, a5
        
        # 更新负数据
        while len(self.negtive_image) > 3000:
            a1 = self.negtive_action.pop(0)
            a2 = self.negtive_image.pop(0)
            a3 = self.negtive_reward.pop(0)
            del a1, a2, a3
    
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

                S.append(self.test_images[game_index][i])
                A.append(self.test_actions[game_index][i])
                R.append(self.test_rewards[game_index][i])

                next_i = min(len(self.test_actions[game_index])-1, i+self.forth_step)
                S_.append(self.test_images[game_index][next_i])
            else:
                i = np.random.choice(len(self.negtive_action), 1)[0]
                S.append(self.negtive_image[i][0])
                S_.append(self.negtive_image[i][1])
                A.append(self.negtive_action[i])
                R.append(self.negtive_reward[i])

        return (tf.convert_to_tensor(np.array(S)), 
                tf.convert_to_tensor(np.array(R)), 
                tf.convert_to_tensor(np.array(A)), 
                tf.convert_to_tensor(np.array(S_)))
    
    def test_sampling(self):
        index = np.random.choice(len(self.test_images), self.batch_size)
        kind = np.random.choice([-1, 0, 1], self.batch_size, p=self.sample_ration)

        result = np.stack([index, kind])
        result = result.transpose()

        return result

    # 获取训练数据
    def getTrainData(self):
        index = self.test_sampling()
        return self.chooseTestData(index)