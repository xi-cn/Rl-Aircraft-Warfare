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


def test(it, avgest, higest):
    high_score = 0
    avg_score = 0
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

            # 选择行为
            action, detail = model.choose_action(x)
            if eps == 0:
                cur_avg = 0
            else:
                cur_avg = int(avg_score / eps)
            print(f"{prefix} eps: {it} it: {eps} greed: [{model.greedy:.2f}] a: {action} detail: [{detail[0]:.2f}, {detail[1]:.2f}, {detail[2]:.2f}] score: [{score}] cur_avg: [{int(cur_avg)}] cur_high: [{high_score}] avg: [{int(avgest)}] high: [{higest}]")

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
                # 添加到数据集
                data_loader.addTestData(
                    np.array(images),
                    np.array(actions),
                    np.array(rewards)
                )
                break
    avg_score /= test_num
    return avg_score, high_score
    

def train(it, avg, high):
    for eps in range(train_step):
        S, R, A, S_ = data_loader.getTrainData()
        model.learn(S, A, R, S_)
        if eps % cover_step == 0:
            model.covergeTargetNet()
            print("target net coverd")
        print(f"model: {prefix} training: iter: [{it}] eps: [{eps}] avg: [{avg:.1f}] high: [{high}]")


def run_maze():
    it = 0
    highest = 0
    avgest = 0

    for i in range(10):
        test(i, 0, 0)
    model.greedy = 0.1
    while True:
        avg, high = test(it, avgest, highest)
        # 写入文件
        with open("../dqn_lstm.txt", 'a', encoding='utf-8') as f:
            f.write(f"{avg} {high}\n")
        print(f"model: {prefix} iter: [{it}] avg: [{avg}] high: [{high}]")
        if avg  > avgest:
            model.save_weights(f"../models/{prefix}_avg.h5")
            avgest = avg
        if high > highest:
            model.save_weights(f"../models/{prefix}_high.h5")
            highest = high
        train(it, avgest, highest)
        model.save_weights(f"../models/{prefix}_last.h5")
        it += 1

if __name__ == "__main__":
    batch_size = 64
    learning_rate = 0.001
    interval=2
    prev_num=20
    forth_step=5
    cover_step=500
    test_num = 50
    train_step=5000
    sample_ratio = (0.1, 0.4, 0.5)

    prefix = "i_2_p20_f5_neg_1"

    model = DQN_LSTM(
        n_actions=3,
    )


    model.target_net.build(input_shape=(None, 11, 160, 160, 3))
    model.eval_net.build(input_shape=(None, 11, 160, 160, 3))
    model.build(input_shape=(None, 11, 160, 160, 3))

    try:
        model.load_weights(f"../models/{prefix}_lastq.h5")
        print("模型权重加载成功")
    except:
        print("模型权重加载失败")

    data_loader = DataWithTest(
        path="../data",
        exist_rate=-1,
        interval=interval,
        prev_num=prev_num,
        forth_step=forth_step,
        batch_size=batch_size,
        sample_ration=sample_ratio
    )

    run_maze()
