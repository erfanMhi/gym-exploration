import gym
import numpy as np

from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.toy_text import discrete

SELF = 0
VERTICAL = 1
HORIZONTAL = 2


class HardSquareEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, exp_value=6):
       
        # Exploration Value
        self.exp_value = exp_value

        # Defining the number of actions and states
        nA = 3
        nS = 4
        
        # Defining the reward system and dynamics of RiverSwim environment
        P, isd = self.__init_dynamics(nS, nA)
        
        super(HardSquareEnv, self).__init__(nS, nA, P, isd)

    def __init_dynamics(self, nS, nA):
        
        # P[s][a] == [(probability, nextstate, reward, done), ...]
        P = {}
        for s in range(nS):
            P[s] = {a: [] for a in range(nA)}

        # Self Transitions
        P[0][SELF] = [(1., 0, 10000./self.exp_value, 0)]
        P[1][SELF] = [(1., 1, 10000.*self.exp_value, 0)]
        P[2][SELF] = [(1., 2, 20000.*self.exp_value, 0)]
        P[3][SELF] = [(1., 3, 20000./self.exp_value, 0)]

        # VERTICAL Transitions
        P[0][VERTICAL] = [(1., 2, -1, 0)]
        P[1][VERTICAL] = [(1., 3, -1, 0)]
        P[2][VERTICAL] = [(1., 0, -1, 0)]
        P[3][VERTICAL] = [(1., 1, -1, 0)]

        # RIGHT Transitions
        P[0][HORIZONTAL] = [(1., 1, 0, 0)]
        P[1][HORIZONTAL] = [(1., 0, 0, 0)]
        P[2][HORIZONTAL] = [(1., 3, -1, 0)]
        P[3][HORIZONTAL] = [(1., 2, -1, 0)]

        # Starting State Distribution
        isd = np.zeros(nS)
        isd[0] = .5
        isd[1] = .5

        return P, isd

    def render(self, mode='human'):
        pass

    def close(self):
        pass
