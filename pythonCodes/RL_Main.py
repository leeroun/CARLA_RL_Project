import glob
import os
import sys
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
EPISODE = 1000
MIN_REWARD = -200

EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001
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
    # Prediction 초기화.
    agent.get_qs([np.ones((env.im_height, env.im_width, 3)), np.ones((env.im_height, env.im_width, 3)),
                  np.ones((env.im_height, env.im_width, 3))])

    print("Start Learning")
    for episode in tqdm(range(1, EPISODE + 1), ascii=True, unit="episodes"):

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
                action = np.random.randint(0, 3)
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
            if episode % 10 == 0:
                print(f'saved model in (models/{RL_Model.MODEL_NAME}_weights_{int(time.time())}')
                agent.model.save_weights(f'models/{RL_Model.MODEL_NAME}_weights_{int(time.time())}')
            # agent.model.save(
            #     f'models/{RL_Model.MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.h5')

        # epsilon 감소
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    agent.terminate = True
    trainer_thread.join()
    print(f'saved model in (models/{RL_Model.MODEL_NAME}_weights_{int(time.time())}')
    agent.model.save_weights(f'models/{RL_Model.MODEL_NAME}_weights_{int(time.time())}')
    env.destroy_actors(True)