import numpy as np
import RL_Model

WEIGHTS_NAME = 'CNN_SEG_weights_1654899490'
agent = RL_Model.DQNAgent()
agent.model.load_weights('models/'+WEIGHTS_NAME)

env = RL_Model.CarEnv(True)
current_state = env.reset()

while True:
    print(agent.get_qs(current_state))
    action = np.argmax(agent.get_qs(current_state))
    print(action)

    current_state, reward, done, _ = env.step(action)
    if done:
        break

for actor in env.actor_list:
    actor.destroy()
