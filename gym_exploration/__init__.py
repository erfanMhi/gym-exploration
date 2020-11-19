from gym.envs.registration import register

register(
    id='HardSquare-v0',
    entry_point='gym_exploration.envs:HardSquareEnv',
    max_episode_steps=20,
)
