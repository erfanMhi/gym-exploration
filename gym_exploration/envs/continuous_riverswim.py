import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class ContinuousRiverswim(gym.Env):
    def __init__(self):
        self.pos = None

        # 0 is right, 1 is left
        mean0, sd0 = -1 / 5, 0.1 / 5
        self.actions = lambda a: (np.random.normal(mean0, sd0)
                                  if a else
                                  (np.random.gamma(342.8, 0.01) - 3.378) / 1.6)
        self.nactions = 2
        
        # NOTE: these aren't hard bounds. The agent can go outside them temporarly
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,)) 
        self.action_space = spaces.Discrete(2)

    def reset(self):
        self.pos = np.random.uniform(0.2, 0.4)
        return self.pos

    def step(self, action):
        assert action == 0 or action == 1
        # move the agent
        self.pos += self.actions(action)

        # calculate rewards
        reward = 1 if self.pos > 1 else 0.0005 if self.pos < 0 else 0
        # reward = self.pos if self.pos > 0.5 else 0 if self.pos > 0 else 0.0005
        # reward = self.pos if self.pos > 0 else 0.0005

        # move agent back in bounds
        if not 0 < self.pos < 1:
            self.pos = min(1, max(0, self.pos))

        terminal = False

        return reward, self.pos, terminal

    def numactions(self):
        return self.nactions
    
    def render(self, mode='human'):
        pass

    def close(self):
        pass


