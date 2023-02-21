import gym
import numpy as np
from rlagents import SarsaQlearning , Qlearning , approximateQlearning , DeepQ


# env = gym.make("LunarLander-v2")
env = gym.make("LunarLander-v2", render_mode="human")

agent = Qlearning(5, flag=True)
agent.train(env)

# sarsaagent = SarsaQlearning(5, flag=True)
# sarsaagent.train(env)

# appxagent = approximateQlearning(100)
# appxagent.train(env)

# deepagent = DeepQ(env.observation_space.shape[0] , env.action_space.n , 1000)
# deepagent.train(env)

env.close()
