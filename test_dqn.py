from DQN_Brain import DQN
import sys
import os
import cv2
import numpy as np

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')  # 使用第二块GPU

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
        env.setTick(10000)
        env.setReward((-5, 1, 2, 3, 0))

        observation, reward, done, score = env.reset()
        env.life_num = 1
        observation = convert_image(observation)

        step = 0

        while True:
            x = observation
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

            step += 1
            # 添加状态
            observation = convert_image(observation)

            if done:
                env.distory()
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
    interval=0
    prev_num=0
    forth_step=2
    cover_step=300
    test_num = 200
    sample_ratio = (0.1, 0.4, 0.5)

    model = DQN(
        n_actions=3,
    )

    model.target_net.build(input_shape=(None, 160, 160, 3))
    model.eval_net.build(input_shape=(None, 160, 160, 3))
    model.build(input_shape=(None, 160, 160, 3))

    try:
        model.load_weights(f"../models/best_dqn.h5")
        print("模型权重加载成功")
    except:
        print("模型权重加载失败")


    average, higest, all_scores = test()
    print(f"average: [{average}]  highest: [{higest}]")

    data = {
        'model': 'dqn',
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


    df = pd.read_csv("../score2.csv", sep=',', index_col=0)
    df.loc[2] = data
    print(df)

    df.to_csv('../score2.csv', sep=',')
