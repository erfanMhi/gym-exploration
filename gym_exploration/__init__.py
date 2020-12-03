from gym.envs.registration import register

register(
    id='HardSquare-v0',
    entry_point='gym_exploration.envs:HardSquareEnv',
    max_episode_steps=20,
)

register(
    id='VarianceWorld-v0',
    entry_point='gym_exploration.envs:VarianceWorld',
)

register(
    id='ContinuousRiverswim-v0',
    entry_point='gym_exploration.envs:ContinuousRiverswim',
)
