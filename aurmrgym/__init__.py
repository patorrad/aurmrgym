from gym.envs.registration import register

register(
    id='arm_training_env_pg-v0',
    entry_point='aurmrgym.envs:TahomaEnv',
    max_episode_steps= None,
)