import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class VarianceWorld(gym.Env):
    def __init__(self):
        self.pos = None

        # 0 is left by N(0.05, 0.01), 1 is right by N(0.05, 0.01)
        self.actions = lambda a: np.random.normal(np.power(-1, 1 - a) * 5) / 100
        self.nactions = 2

        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=(1,))
        self.action_space = spaces.Discrete(2)

        self.noisy_reward = lambda: np.random.choice([2, -2, 0.1, -0.1, 1])
     

    def reset(self):
        self.pos = np.random.uniform(0.45, 0.55)
        return self.pos

    def render(self, mode='human'):
        pass

    def step(self, action):
        self.pos += self.actions(action)

        noisy = self.pos > 1
        stable = self.pos < 0

        terminal = noisy or stable
        reward = self.noisy_reward() if noisy else 0.02 if stable else 0

        return reward, self.pos, terminal

    def numactions(self):
        return self.nactions
