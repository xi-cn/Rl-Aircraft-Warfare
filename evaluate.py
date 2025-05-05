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


# 用于接收进程信息的线程
def receive_worker(remote, t_bar, scores, rewards, estimate_q, true_q):
    try:
        for test_time in range(test_num // num_workers):
            if exit_event.is_set():
                return
            try:
                estimate_value, true_reward, score, reward = remote.recv()
            except (EOFError, ConnectionResetError):
                return  # 管道关闭时退出

            try:
                lock.acquire()
                scores.append(score)
                rewards.append(reward)
                r = 0
                for i in range(len(true_reward)):
                    k = len(true_reward)-1 - i
                    r = r * config['gamma'] + true_reward[k]
                    true_reward[k] = r
                estimate_q.extend(estimate_value)
                true_q.extend(true_reward)

                t_bar.set_postfix(
                    OrderedDict([
                        ('R avg', int(sum(rewards)/len(rewards)*10)/10),
                        ('S avg', int(sum(scores) / len(scores))),
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
        # 创建测试子进程
        p = Process(target=worker, args=(work_remote, MODEL, config, True))
        p.start()
        processes.append(p)


    # 发送参数
    cpu_model.set_weights(model.get_weights())
    for remote in remotes:
        remote.send((cpu_model.get_weights(), None))


    # 所有进程游戏的分数和奖励
    scores, rewards, estimate_q, true_q = [], [], [], []
    t_bar = tqdm(desc="测试游戏", total=test_num // num_workers * num_workers)

    # 创建多线程用于接收进程信息
    threads = []
    for remote in remotes:
        t = threading.Thread(target=receive_worker, args=(remote, t_bar, scores, rewards, estimate_q, true_q), daemon=True)
        threads.append(t)
        t.start()
    # 等待线程结束
    for t in threads:
        t.join()

    t_bar.close()
    # 整理得分和奖励
    with open(f"./results/{model_name}/evaluate.txt", 'w', encoding='utf-8') as f:
        for score, reward in zip(scores, rewards):
            f.write(f'{score} {reward}\n')
    with open(f"./results/{model_name}/q_value.txt", 'w', encoding='utf-8') as f:
        for e, t in zip(estimate_q, true_q):
            f.write(f'{e} {t}\n')




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
    
    config['epochs'] = 1
    config['test_num'] = 50
    config['num_workers'] = 1

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

    # 选择模型
    if model_name == 'DQN':
        MODEL = DQN
    elif model_name == "DDQN":
        MODEL = DDQN
    elif model_name == "DuelingDQN":
        MODEL = DuelingDQN

    # 创建cpu模型副本
    with tf.device('CPU'):
        cpu_model = MODEL(n_actions, gamma, learning_rate)
    model = MODEL(n_actions, gamma, learning_rate)
    # 构建模型
    cpu_model.build(input_shape=(None, prev_num+1, 160, 160, 3))
    model.build(input_shape=(None, prev_num+1, 160, 160, 3))

    # 加载模型权重
    model.load_weights(f'./results/{model_name}/avg.h5')


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