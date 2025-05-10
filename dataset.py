import numpy as np
import cv2

# 对图片预处理
def convert_image(observation:np.ndarray):
    # 通道转化
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    # 图片缩放
    observation = cv2.resize(observation, (160, 160)).astype(np.float16)
    # 归一化
    observation = observation / 255.0
    return observation


class LSTM_Dataset:
    def __init__(
        self,
        interval=2,
        prev_num=10,
        sample_ration = (0.3, 0.4, 0.3),
        forth_step=5,
        batch_size=64,
        max_test_num=300000,
        max_neg_num=10000
    ):
        self.interval=interval
        self.prev_num=prev_num
        self.forth_step=forth_step
        self.max_test_num=max_test_num
        self.max_neg_num=max_neg_num
        self.batch_size=batch_size
        self.prev = prev_num * interval
        
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

        if len(rewards) < self.prev + 10:
            return
        
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
                # 最后一步没有next
                self.negtive_image.append((state, state))
                self.negtive_action.append(np.copy(actions[i]))
                self.negtive_reward.append(np.copy(rewards[i]))
                break

        
        if len(test_zero) == 0:
            return

        self.test_positive.append(np.array(test_positive))
        self.test_zero.append(np.array(test_zero))
        self.test_images.append(images)
        self.test_actions.append(actions)
        self.test_rewards.append(rewards)
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
        
        # 维护负样本数量
        while len(self.negtive_image) > self.max_neg_num:
            self.negtive_action.pop(0)
            self.negtive_image.pop(0)
            self.negtive_reward.pop(0)


    
    # 获取训练的数据
    def chooseTestData(self, index):
        S = []
        A = []
        R = []
        S_ = []

        for (game_index, kind) in index:
            # 非负样本
            if not kind == -1:
                if kind == 0:
                    i = np.random.choice(self.test_zero[game_index], 1)[0]
                # 防止无正样本
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
                
            # 负样本
            else:
                i = np.random.choice(len(self.negtive_action), 1)[0]
                S.append(self.negtive_image[i][0])
                S_.append(self.negtive_image[i][1])
                A.append(self.negtive_action[i])
                R.append(self.negtive_reward[i])
                shape = self.negtive_image[i][1].shape

                    
        S=np.array(S)
        S_=np.array(S_)
        R=np.array(R)
        A=np.array(A)

        return S, R, A, S_

    # 测试数据采样
    def test_sampling(self):
        # 采样游戏的场次
        index = np.random.choice(len(self.test_images), self.batch_size)
        # 对应场次选择 样本类别
        kind = np.random.choice([-1, 0, 1], self.batch_size, p=self.sample_ration)

        result = np.stack([index, kind])
        result = result.transpose()

        return result

    # 获取训练数据
    def getTrainData(self):
        # 采样
        index = self.test_sampling()
        return self.chooseTestData(index)