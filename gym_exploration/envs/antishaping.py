import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from scipy.stats import norm

class Antishaping(gym.Env):
    """
    Note: requires gamma to be set to maintain reward structure.
    """
    def __init__(self, gamma=0.99):
        self.pos = None
        self.terminal_reward = (norm.pdf(0.5, 0.5, 0.15)
                                / (1 - gamma)
                                / np.power(gamma, 100)  # 100 steps to edge
                                * 1.005)

        # 0 is left by N(0.005, 0.001), 1 is right by N(0.005, 0.001)
        self.actions = lambda a: (np.random.normal(np.power(-1, 1 - a) * 5) /
                                  1000)
        self.nactions = 2

        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=(1,))
        self.action_space = spaces.Discrete(2)


    def reset(self):
        self.pos = np.random.uniform(0.45, 0.55)
        return self.pos

    def step(self, action):
        assert action == 0 or action == 1

        self.pos += self.actions(action)

        terminal = not 0 < self.pos < 1
        reward = norm._pdf((self.pos - 0.5) / 0.15) / 0.15

        # normalize to 1 terminal reward
        reward /= self.terminal_reward
        reward += 1 if terminal else 0

        return reward, self.pos, terminal

    def numactions(self):
        return self.nactions
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass
