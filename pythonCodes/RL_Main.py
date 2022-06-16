import glob
import logging
import os
import sys
import traceback

import RL_Model

try:
    sys.path.append(glob.glob('../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' %
                              (
                                  sys.version_info.major,
                                  sys.version_info.minor,
                                  'win-amd64'
                              ))[0])
except IndexError:
    pass

import tensorflow as tf
import random
import time
import numpy as np
import keras.backend as backend
from threading import Thread

from tqdm import tqdm

MEMORY_FRACTION = 0.8
AGGREGATE_STATS_EVERY = 10
EPISODE = 5000
MIN_REWARD = -200

EPSILON_DECAY = 0.9999
MIN_EPSILON = 0.05
epsilon = 1

if __name__ == "__main__":
    print("Initializing")
    FPS = 60
    ep_rewards = [-200]

    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    # Multi Agent를 위한 메모리 fraction
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

    if not os.path.isdir("models"):
        os.makedirs("models")

    print("Create Agent & Environment")
    # Agent / Environment 생성
    agent = RL_Model.DQNAgent()
    env = RL_Model.CarEnv()

    # Thread 시작
    trainer_thread = Thread(target=agent.train_in_loop)
    trainer_thread.start()

    while not agent.training_initialized:
        time.sleep(0.01)

    print("Prediction 초기화")

    # left_dis (1) right_dis (1) delta angle (1) left lane type (4) right lane type (4)
    # Accel (1) Gyro (3)
    # Dis (3) [Left Front Right]
    qs_left_dis = np.ones(1).astype(np.float32)
    qs_right_dis = np.ones(1).astype(np.float32)
    qs_delta_angle = np.ones(1).astype(np.float32)
    qs_left_type = [1, 0, 0, 0]
    qs_right_type = [1, 0, 0, 0]
    qs_accel = np.ones(1).astype(np.float32)
    qs_gyro = 1
    qs_dis = [1, 1, 1]

    agent.get_qs([qs_left_dis, qs_right_dis, qs_delta_angle, qs_left_type, qs_right_type, qs_accel, qs_gyro, qs_dis])

    print("Start Learning")
    for episode in tqdm(range(1, EPISODE + 1), ascii=True, unit="episodes"):
        try:
            print(f"Episode {episode} / {EPISODE}")
            env.collision_hist = []

            # Tensor Board 업데이트
            agent.tensorboard.step = episode

            # Episode 초기화
            episode_reward = 0
            step = 1

            # Env 초기화
            current_state = env.reset()
            done = False
            episode_start = time.time()

            # Step 진행
            while True:
                if np.random.random() > epsilon:  # epsilon 이상이면 기존 table 값 사용
                    action = np.argmax(agent.get_qs(current_state))
                    print(agent.get_qs(current_state))
                else:  # epsilon 이하면 랜덤으로
                    action = np.random.randint(0, RL_Model.ACTION_NUMBER)
                time.sleep(1 / FPS)
                new_state, reward, done, _ = env.step(action)
                episode_reward += reward

                # replay_memory에 저장
                agent.update_replay_memory((current_state, action, reward, new_state, done))
                step += 1

                current_state = new_state

                if done:
                    break

            env.destroy_actors()
            print(f"episode reward {episode_reward}")
            # reward 저장
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                               epsilon=epsilon)

                # 모델 저장
                if episode % 100 == 0:
                    try:
                        print(f'saved model in (models/{RL_Model.MODEL_NAME}_episode_{episode}')
                        agent.model.save_weights(f'models/{RL_Model.MODEL_NAME}_episode_{episode}')
                    except OSError:
                        print(OSError)
                        print(f'save model fail in episode {episode}')
                # agent.model.save(
                #     f'models/{RL_Model.MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.h5')

            # epsilon 감소
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

        except Exception as e:
            logging.error(traceback.format_exc())
            env.destroy_actors()

    agent.terminate = True
    trainer_thread.join()
    print(f'saved model in (models/{RL_Model.MODEL_NAME}_weights_{int(time.time())}')
    agent.model.save_weights(f'models/{RL_Model.MODEL_NAME}_weights_{int(time.time())}')

    with open("file.txt", 'w') as f:
        for reward in ep_rewards:
            f.write(str(reward) + '\n')
