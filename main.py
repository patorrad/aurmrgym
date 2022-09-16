
import gym
from gym.spaces import Discrete, Box, Dict
import aurmrgym

def main(env_name, lr):
    # env = gym.make(env_name)
    env = gym.make('arm_training_env_pg-v0')
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Box), \
        "This example only works for envs with discrete action spaces."

    # PyTorch needs to be imported before isaacgym modules. 
    import test_env_pg
    test_env_pg.train(env, lr=lr)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='arm_training_env_pg-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    main(env_name=args.env_name, lr=args.lr)
