import sys
import os
import cv2
import numpy as np
import pygame
from time import sleep
import os


sys.path.append("./game_env")
current_dir = os.path.dirname(os.path.abspath(__file__))
env_dir = os.path.join(current_dir, "game_env")

# 将环境路径导入到系统路径中
sys.path.append(env_dir)

from game_env.maze_env_play import Maze

# 切换工作目录
os.chdir(env_dir)

# 对图片预处理
def convert_image(observation:np.ndarray):
    # 通道转化
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    # 图片缩放
    observation = cv2.resize(observation, (160, 160))
    # 图片旋转
    observation = np.transpose(observation, (1, 0, 2))

    return observation
    


def run_maze():

    if len(os.listdir("../new_data")) == 0:
        files = os.listdir("../data")
    else:
        files = os.listdir("../new_data")
        
    if len(files) == 0:
        episode = 0
    else:
        f = files[-1].split("_")
        episode = int(f[0]) + 1
    print(episode)


    num = 0
    while True:

        q = input()
        sleep(1)

        # initial observation
        observation, reward, done, score, action = env.reset()
        env.life_num = 1
        observation = convert_image(observation)

        env.setTick(80)
        step = 0
        while True:
            num += 1
            if num % 100 == 0:
                print(num)

            # RL take action and get next observation and reward
            observation_, reward, done, score, action = env.step()
            if score > 150000:
                env.setTick(60)
            # 保存图像
            cv2.imwrite(f"../new_data/{episode:04d}_{step:07d}_{action}_{reward}.png", observation)

            # 图像转化
            observation_ = convert_image(observation_)
            # 更新图片
            observation = observation_

            step += 1

            if done:
                # 保存最后一帧
                # 保存图像
                cv2.imwrite(f"../new_data/{episode:04d}_{step:07d}_{action}_{1000}.png", observation)
                # 记录得分
                with open("../score.txt", 'a', encoding='utf-8') as f:
                    f.write(f"{score}\n")
                step += 1
                break

        episode += 1
        

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":

    env = Maze()
    env.setStepBatchs(3)
    env.setTick(100)
    env.setReward((-1, 1, 2, 3))

    run_maze()
