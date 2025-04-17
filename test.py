from TensorflowAgent import DQN_LSTM, DDPG
from TensorflowAgent import DataWithTest
import sys
import os
import cv2
import numpy as np


import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')  # 使用第一块GPU

# # 禁用GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


sys.path.append("./game_env")
current_dir = os.path.dirname(os.path.abspath(__file__))
env_dir = os.path.join(current_dir, "game_env")
# 将环境路径导入到系统路径中
sys.path.append(env_dir)

from game_env.maze_env import Maze
# 切换工作目录
os.chdir(env_dir)

# 对图片预处理
def convert_image(observation:np.ndarray):
    # 通道转化
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    # 图片缩放
    observation = cv2.resize(observation, (160, 160)).astype(np.float16)
    # 归一化
    observation = observation / 255.0
    return observation


def test():
    high_score = 0
    avg_score = 0
    all_scores = []
    for eps in range(test_num):
        # maze game
        env = Maze()
        env.setStepBatchs(2)
        env.setTick(5000)
        env.setReward((-5, 1, 2, 3, 0))

        observation, reward, done, score = env.reset()
        env.life_num = 1
        observation = convert_image(observation)

        step = 0
        images = []
        actions = []
        rewards = []

        while True:
            images.append(observation)

            # 添加输入集合
            x = []
            low_index = max(0, step-interval*prev_num)
            x = images[low_index:step+1:interval]
            x = np.expand_dims(x, axis=0)


            model.act_greedy = True
            model.step = 100
            # 选择行为
            action, detail = model.choose_action(x)
            if eps == 0:
                avg = 0
            else:
                avg = int(avg_score/eps)
            print(f"eps: [{eps}] action: [{action}] detail: {detail} score: [{score}] avg: [{avg}] high: [{high_score}]")

            # 执行行为
            observation, reward, done, score = env.step(action)
            actions.append(action)
            rewards.append(reward)

            step += 1
            # 添加状态
            observation = convert_image(observation)

            if done:
                env.distory()
                images.append(observation)
                actions.append(action)
                rewards.append(reward)
                if score > high_score:
                    high_score = score
                avg_score += score
                all_scores.append(str(score))
                break

    avg_score /= test_num
    return avg_score, high_score, all_scores
    


import pandas as pd


if __name__ == "__main__":
    batch_size = 32
    learning_rate = 0.001
    interval=2
    prev_num=20
    forth_step=5
    cover_step=500
    test_num = 200
    sample_ratio = (0.1, 0.4, 0.5)

    model = DQN_LSTM(
        n_actions=3,
    )

    model.target_net.build(input_shape=(None, 11, 160, 160, 3))
    model.eval_net.build(input_shape=(None, 11, 160, 160, 3))
    model.build(input_shape=(None, 11, 160, 160, 3))

    try:
        model.load_weights(f"../models/best_lstm_dqn.h5")
        print("模型权重加载成功")
    except:
        print("模型权重加载失败")


    average, higest, all_scores = test()
    print(f"average: [{average}]  highest: [{higest}]")

    data = {
        'model': 'dqn_lstm',
        'interval': interval,
        'prev_num': prev_num,
        'forth_step': forth_step,
        'cover_step': cover_step,
        'negtive_ratio': sample_ratio[0],
        'positive_ratio': sample_ratio[2],
        'average': average,
        'higest': higest,
        'scores': ' '.join(all_scores)
    }


    df = pd.read_csv("../score_up.csv", sep=',', index_col=0)
    df.loc[3] = data
    print(df)

    df.to_csv('../score_36.csv', sep=',')
