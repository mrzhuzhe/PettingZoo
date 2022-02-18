#from pettingzoo.butterfly import pistonball_v6
#env = pistonball_v6.env()
from pettingzoo.butterfly import prospector_v4
env = prospector_v4.env()

def policy(agent, observation):
    #print(agent)
    return env.action_space(agent).sample()

env.reset()
done = False

for agent in env.agent_iter():
    env.render()
    observation, reward, done, info = env.last()
    if not done:
        action = policy(agent, observation)
        env.step(action)
    else:
        env.step(None)