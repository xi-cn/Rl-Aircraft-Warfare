import sys
import os
import numpy as np
from dataset import convert_image


import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')  # 使用第一块GPU
    tf.config.experimental.set_memory_growth(gpus[0], True)


def worker(remote, MODEL, config):
    # 导入游戏运行环境
    sys.path.append("./game_env")
    from game_env.maze_env import Maze

    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_dir = os.path.join(current_dir, "game_env")
    os.chdir(env_dir)

    # 创建模型
    model = MODEL(
        n_actions = config['n_actions'],
        gamma = config['gamma'],
        learning_rate = config['learning_rate']
    )
    model.build(input_shape=(None, config['prev_num']+1, 160, 160, 3))


    for epoch in range(config['epochs']):
        # 接收模型参数
        weights, _ = remote.recv()
        model.set_weights(weights)

        # 测试环境
        for it in range(config['test_num'] // config['num_workers']):

            # 初始化游戏环境
            env = Maze()
            env.setStepBatchs(2)
            env.setTick(5000)
            env.setReward(config['reward_config'])

            observation, reward, done, score = env.reset()
            env.life_num = 1
            observation = convert_image(observation)
            # 存储游戏中间变量
            step = 0
            images = []
            actions = []
            rewards = []

            score_sum = 0
            reward_sum = 0
            # 进行一局游戏
            while True:
                images.append(observation)

                # 添加输入集合
                x = []
                low_index = max(0, step-config['interval']*config['prev_num'])
                x = images[low_index:step+1:config['interval']]
                x = np.expand_dims(x, axis=0)

                # 选择行为
                action, detail = model.choose_action(x)

                # 执行行为
                observation, reward, done, score = env.step(action)
                actions.append(action)
                rewards.append(reward)

                score_sum = score
                reward_sum += reward

                step += 1
                # 添加状态
                observation = convert_image(observation)
                # 游戏结束
                if done:
                    env.distory()
                    images.append(observation)
                    actions.append(action)
                    rewards.append(reward)
                    break
            images, actions, rewards = np.array(images), np.array(actions), np.array(rewards)
            # 发送数据到主进程
            remote.send((images, actions, rewards, score_sum, reward_sum))


def start_worker(remote, MODEL, config):
    # 导入游戏运行环境
    sys.path.append("./game_env")
    from game_env.maze_env import Maze

    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_dir = os.path.join(current_dir, "game_env")
    os.chdir(env_dir)

    # 创建模型
    model = MODEL(
        n_actions = config['n_actions'],
        gamma = 0
    )
    model.greedy = 0

    while True:
        # 初始化游戏环境
        env = Maze()
        env.setStepBatchs(2)
        env.setTick(5000)
        env.setReward(config['reward_config'])

        observation, reward, done, score = env.reset()
        env.life_num = 1
        observation = convert_image(observation)
        # 存储游戏中间变量
        step = 0
        images = []
        actions = []
        rewards = []

        # 进行一局游戏
        while True:
            images.append(observation)

            # 添加输入集合
            x = []
            low_index = max(0, step-config['interval']*config['prev_num'])
            x = images[low_index:step+1:config['interval']]
            x = np.expand_dims(x, axis=0)

            # 选择行为
            action, detail = model.choose_action(x)

            # 执行行为
            observation, reward, done, score = env.step(action)
            actions.append(action)
            rewards.append(reward)

            step += 1
            # 添加状态
            observation = convert_image(observation)
            # 游戏结束
            if done:
                env.distory()
                images.append(observation)
                actions.append(action)
                rewards.append(reward)
                break
        images, actions, rewards = np.array(images), np.array(actions), np.array(rewards)
        # 发送数据到主进程
        remote.send((images, actions, rewards))
        # 判断是否结束
        is_over, _ = remote.recv()
        if is_over:
            break
