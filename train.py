import sys
import os
import cv2
import numpy as np
from model import DQN, DDQN, DuelingDQN, RainbowDQN
from dataset import convert_image, LSTM_Dataset
import argparse
import json


import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')  # 使用第一块GPU

# # 禁用gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 将环境路径导入到系统路径中
sys.path.append("./game_env")
current_dir = os.path.dirname(os.path.abspath(__file__))
env_dir = os.path.join(current_dir, "game_env")
sys.path.append(env_dir)

from game_env.maze_env import Maze
# 切换工作目录
os.chdir(env_dir)

# 测试模型
def test(epoch, avgest, higest):
    # 最高分 平均分 奖励总和
    high_score = 0
    avg_score = 0
    tot_reward = 0
    for iter in range(test_num):
        # 初始化游戏环境
        env = Maze()
        env.setStepBatchs(2)
        env.setTick(5000)
        env.setReward(reward_config)

        observation, reward, done, score = env.reset()
        env.life_num = 1
        observation = convert_image(observation)
        # 存储游戏中间变量
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

            # 选择行为
            action, detail = model.choose_action(x)
            if iter == 0:
                cur_avg = 0
            else:
                cur_avg = int(avg_score / iter)
            print(f"{model_name} epoch: {epoch} it: {iter} greed: [{model.greedy:.2f}] a: {action} detail: [{detail[0]:.2f}, {detail[1]:.2f}, {detail[2]:.2f}] score: [{score}] cur_avg: [{int(cur_avg)}] cur_high: [{high_score}] avg: [{int(avgest)}] high: [{higest}]")

            # 执行行为
            observation, reward, done, score = env.step(action)
            actions.append(action)
            rewards.append(reward)
            tot_reward += reward

            step += 1
            # 添加状态
            observation = convert_image(observation)
            # 游戏结束
            if done:
                env.distory()
                images.append(observation)
                actions.append(action)
                rewards.append(reward)
                if score > high_score:
                    high_score = score
                avg_score += score
                # 添加到数据集
                data_loader.addTestData(
                    np.array(images),
                    np.array(actions),
                    np.array(rewards)
                )
                break
    avg_score /= test_num
    return avg_score, high_score, tot_reward / test_num

# 训练模型
def train(epoch, avg, high):
    for step in range(train_step):
        S, R, A, S_ = data_loader.getTrainData()
        model.learn(S, A, R, S_)
        # 覆盖网络
        if step % cover_step == 0:
            model.covergeTargetNet()
            print("target net coverd")
        print(f"model: {model_name} training: epoch: [{epoch}] step: [{step}] avg: [{avg:.0f}] high: [{high}]")


# 运行环境 训练模型
def run_maze():
    it = 0
    highest = 0
    avgest = 0

    # 先随机测试 增加初始数据集
    for i in range(1):
        test(i, 0, 0)

    while True:
        # 测试模型 获取数据集
        avg, high, reward = test(it, avgest, highest)
        # 测试分数
        with open(f"../results/{model_name}/record.txt", 'a', encoding='utf-8') as f:
            f.write(f"{avg} {high} {reward}\n")
        print(f"model: {model_name} iter: [{it}] avg: [{avg}] high: [{high}]")
        
        # 保存最好的模型
        if avg  > avgest:
            if it > 0:
                model.save_weights(f"../results/{model_name}/avg.h5")
            avgest = avg
        if high > highest:
            if it > 0:
                model.save_weights(f"../results/{model_name}/high.h5")
            highest = high
        
        # 训练模型
        train(it, avgest, highest)
        model.save_weights(f"../results/{model_name}/last.h5")
        it += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 参数
    parser.add_argument('-config', help='配置文件名称')
    # 解析参数
    args = parser.parse_args()

    config_file = args.config
    # 读取配置文件
    with open('../configs/{}.json'.format(config_file), 'r') as f:
        config = json.load(f)

    model_name = config['model']
    n_actions = config['n_actions']
    gamma = config['gamma']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    interval = config['interval']
    prev_num = config['prev_num']
    forth_step = config['forth_step']
    cover_step = config['cover_step']
    test_num = config['test_num']
    train_step = config['train_step']
    sample_ratio = config['sample_ratio']
    reward_config = config['reward_config']
    max_test_num = config['max_test_num']
    max_neg_num = config['max_neg_num']

    if not os.path.exists(f'../results/{model_name}'):
        os.mkdir(f'../results/{model_name}')

    # 数据集加载器
    data_loader = LSTM_Dataset(
        interval=interval,
        prev_num=prev_num,
        sample_ration=sample_ratio,
        forth_step=forth_step,
        batch_size=batch_size,
        max_test_num=max_test_num,
        max_neg_num=max_neg_num
    )

    # 配置模型
    if model_name == 'DQN':
        model = DQN(
            n_actions,
            gamma,
            learning_rate
        )
    elif model_name == "DDQN":
        model = DDQN(
            n_actions,
            gamma,
            learning_rate
        )

    with open(f"../results/{model_name}/record.txt", 'w', encoding='utf-8') as f:
        pass

    run_maze()