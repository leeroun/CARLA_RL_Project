import numpy as np
import RL_Model

WEIGHTS_NAME = 'DNN_V1_weights_1655128651'
agent = RL_Model.DQNAgent()
agent.model.load_weights('models/'+WEIGHTS_NAME)

env = RL_Model.CarEnv()
current_state = env.reset()

while True:
    # Env 초기화
    current_state = env.reset()
    done = False
    t = 0
    while True:
        if np.random.random() > 0.2:  # epsilon 이상이면 기존 table 값 사용
            print(agent.get_qs(current_state))
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, RL_Model.ACTION_NUMBER)
        print(action)

        current_state, reward, done, _ = env.step(action)
        if done:
            break
        t += 1
    env.destroy_actors()

