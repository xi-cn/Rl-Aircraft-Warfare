import sys
import os
import numpy as np
import argparse
import yaml
import tensorflow as tf
from multiprocessing import Process, Pipe, Lock
from tqdm import tqdm
from collections import OrderedDict
import threading
# 线程锁
lock = threading.Lock()
# 全局退出标志
exit_event = threading.Event()


from model import DQN, DDQN, DuelingDQN, RainbowDQN
from dataset import convert_image, LSTM_Dataset
from work import worker, start_worker
import multiprocessing as mp

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # 动态分配内存
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)


# 训练模型
def train(epoch, avg, high, reward):
    for step in range(train_step):
        S, R, A, S_ = data_loader.getTrainData()
        loss = model.learn(S, A, R, S_)
        # 覆盖网络
        if (step + 1) % cover_step == 0:
            model.covergeTargetNet()
            print("target net coverd")
        if (step + 1) % 10 == 0:
            print(f"{model_name} : epoch: [{epoch}] step: [{step+1}] loss: [{loss:.5f}] avg: [{avg:.0f}] high: [{high}] reward: [{reward:.0f}]")

# 用于接收进程信息的线程
def receive_worker(remote, t_bar, scores, rewards, avg_reward, avg_score):
    try:
        for test_time in range(test_num // num_workers):
            if exit_event.is_set():
                return
            try:
                s, a, r, score, reward = remote.recv()
            except (EOFError, ConnectionResetError):
                return  # 管道关闭时退出

            try:
                lock.acquire()
                data_loader.addTestData(s, a, r)
                scores.append(score)
                rewards.append(reward)
                t_bar.set_postfix(
                    OrderedDict([
                        ('R avg', int(sum(rewards)/len(rewards)*10)/10),
                        ('Hist R', int(avg_reward)),
                        ('S avg', int(sum(scores) / len(scores))),
                        ('Hist S', int(avg_score)),
                    ])
                )
            finally:
                lock.release()
            t_bar.update(1)
    except Exception as e:
        print(f"线程异常: {e}")



# 运行
def run():
    global processes
    global remotes

    # 创建管道
    remotes, work_remotes = zip(*[Pipe() for _ in range(num_workers)])
    for i, work_remote in enumerate(work_remotes):
        # 创建启动子进程
        p = Process(target=start_worker, args=(work_remote, MODEL, config))
        p.start()
        processes.append(p)

    one_pre_test = pre_test_num // num_workers
    t_bar = tqdm(total=one_pre_test * num_workers, desc="收集启动数据集")
    # 收集启动数据
    for pre_test in range(one_pre_test):
        # 接收单局游戏
        for i, remote in enumerate(remotes):

            # 单局游戏数据集
            s, a, r = remote.recv()
            # 将数据添加到数据加载器中
            data_loader.addTestData(s, a, r)
            # 发送是否结束信号
            remote.send((pre_test == one_pre_test-1, None))
            t_bar.update(1)

    t_bar.close()
    # 结束子进程
    for p in processes:
        p.join()
    processes = []

    for i, work_remote in enumerate(work_remotes):
        # 创建测试子进程
        p = Process(target=worker, args=(work_remote, MODEL, config))
        p.start()
        processes.append(p)

    avg_score = 0
    high_score = 0
    avg_reward = -1000
    # 循环训练
    for epoch in range(epochs):
        # 训练模型
        train(epoch, avg_score, high_score, avg_reward)

        # 发送参数
        cpu_model.set_weights(model.get_weights())
        for remote in remotes:
            remote.send((cpu_model.get_weights(), None))


        # 所有进程游戏的分数和奖励
        scores, rewards = [], []
        t_bar = tqdm(desc="测试游戏", total=test_num // num_workers * num_workers)

        # 创建多线程用于接收进程信息
        threads = []
        for remote in remotes:
            t = threading.Thread(target=receive_worker, args=(remote, t_bar, scores, rewards, avg_reward, avg_score), daemon=True)
            threads.append(t)
            t.start()
        # 等待线程结束
        for t in threads:
            t.join()

        t_bar.close()
        # 整理得分和奖励
        avg_s = sum(scores) / len(scores)
        high_s = max(scores)
        avg_r = sum(rewards) / len(rewards)
        
        # 测试分数
        with open(f"./results/{model_name}/record.txt", 'a', encoding='utf-8') as f:
            f.write(f"{avg_s} {high_s} {avg_r}\n")
        print(f"model: {model_name} epoch: [{epoch}] avg: [{avg_s}] high: [{high_s}]  avg reward: [{avg_r}]")
        
        # 保存最好的模型
        if avg_s  > avg_score:
            model.save_weights(f"./results/{model_name}/avg.h5")
            avg_score = avg_s
        if high_s > high_score:
            model.save_weights(f"./results/{model_name}/high.h5")
            high_score = high_s
        avg_reward = max(avg_reward, avg_r)




if __name__ == "__main__":
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    # 参数
    parser.add_argument('-config', help='配置文件名称')
    # 解析参数
    args = parser.parse_args()

    config_file = args.config
    # 读取配置文件
    with open('./configs/{}.yaml'.format(config_file), 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model']
    n_actions = config['n_actions']
    gamma = config['gamma']
    batch_size = config['batch_size']
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    interval = config['interval']
    prev_num = config['prev_num']
    forth_step = config['forth_step']
    cover_step = config['cover_step']
    test_num = config['test_num']
    train_step = config['train_step']
    pre_test_num = config["pre_test_num"]
    sample_ratio = config['sample_ratio']
    reward_config = config['reward_config']
    max_test_num = config['max_test_num']
    max_neg_num = config['max_neg_num']

    num_workers = config['num_workers']

    if not os.path.exists(f'./results/{model_name}'):
        os.mkdir(f'./results/{model_name}')

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

    # 选择模型
    if model_name == 'DQN':
        MODEL = DQN
    elif model_name == "DDQN":
        MODEL = DDQN

    # 创建cpu模型副本
    with tf.device('CPU'):
        cpu_model = MODEL(n_actions, gamma, learning_rate)
    model = MODEL(n_actions, gamma, learning_rate)
    # 构建模型
    cpu_model.build(input_shape=(None, prev_num+1, 160, 160, 3))
    model.build(input_shape=(None, prev_num+1, 160, 160, 3))


    with open(f"./results/{model_name}/record.txt", 'w', encoding='utf-8') as f:
        pass

    # 子进程列表
    global processes
    global remotes

    processes = []
    remotes = []
    try:
        run()
    except KeyboardInterrupt:
        exit_event.set()
        # 关闭管道
        for remote in remotes:
            remote.close()
        # 终止进程
        for p in processes:
            p.terminate()
            p.join(timeout=1.0)
            if p.is_alive():
                p.kill()

        sys.exit(0)