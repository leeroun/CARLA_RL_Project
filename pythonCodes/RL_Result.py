import numpy as np
import RL_Model

WEIGHTS_NAME = 'DNN_V3_l5_A5_episode_1500'
agent = RL_Model.DQNAgent()
agent.model.load_weights('models/'+WEIGHTS_NAME)

env = RL_Model.CarEnv()
current_state = env.reset()

qs_left_dis = np.ones(1).astype(np.float32)
qs_right_dis = np.ones(1).astype(np.float32)
qs_delta_angle = np.ones(1).astype(np.float32)
qs_left_type = [1, 0, 0, 0]
qs_right_type = [1, 0, 0, 0]
qs_accel = np.ones(1).astype(np.float32)
qs_gyro = 1
qs_dis = [1, 1, 1]

agent.get_qs([qs_left_dis, qs_right_dis, qs_delta_angle, qs_left_type, qs_right_type, qs_accel, qs_gyro, qs_dis])

while True:
    # Env 초기화
    current_state = env.reset()
    done = False
    t = 0
    while True:
        action = np.argmax(agent.get_qs(current_state))
        print(agent.get_qs(current_state))

        current_state, reward, done, _ = env.step(action)
        if done:
            break
        t += 1
    env.destroy_actors()

