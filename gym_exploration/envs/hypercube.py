import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class Hypercube(gym.Env):
    """
    Note: requires gamma to be set to maintain reward structure.
    """
    def __init__(self, n=5, gamma=0.99):
        self.pos = None
        self.dim = n
        self.radius = 10

        # 0 is left, or N(-1, 0.15), 1 is right, or N(+1, 0.15)
        self.actions = lambda a: np.random.normal(np.power(-1, 1 - a), 0.15)
        self.nactions = 2 * n

        # reward code
        self.rewards = np.zeros(self.dim + 1)
        self.rewards[-1] = 1
        for i in reversed(range(1, self.dim)):
            self.rewards[i] = self.rewards[i + 1] * (1 - gamma) * 0.9
        # make average reward 100x higher when agent chooses to receive final
        # terminating reward
        self.rewards[-1] = (111 * self.rewards[-2] * self.radius * self.dim
                            - (self.rewards[1:self.dim] * 10).sum())

        self.observation_space = spaces.Box(low=-self.radius, high=self.radius, shape=(self.dim,))
        self.action_space = spaces.Discrete(2)

    def reset(self):
        self.pos = np.zeros(self.dim)
        return self.pos

    def step(self, action):
        assert action == 0 or action == 1

        # make the action vector from the integer action selected
        action_dim = action // 2
        action_dir = action % 2
        action_vector = np.zeros(self.dim)
        action_vector[action_dim] = self.actions(action_dir)

        # move the agent and make sure it stays within bounds
        self.pos += action_vector
        if abs(self.pos[action_dim]) > self.radius:
            self.pos[action_dim] = min(self.radius,
                                       max(-self.radius,
                                           self.pos[action_dim]))

        num_walls = (np.abs(self.pos) == self.radius).sum()
        terminal = num_walls == self.dim
        reward = self.rewards[num_walls]

        return self.pos, reward, terminal, {}

    def numactions(self):
        return self.nactions

    def render(self, mode='human'):
        pass

    def close(self):
        pass
