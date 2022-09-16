import torch
import torch.nn as nn
from torch.distributions.continuous_bernoulli import ContinuousBernoulli
from torch.optim import Adam
import numpy as np
import copy

# These are points in a set of bins
BIN_X_COORD = [-0.1, 0.15, 0.35, 0.6]
BIN_Y_COORD = [1, 1.3, 1.45, 1.6, 1.74, 1.95, 2.08]
BIN_Z_COORD = [-0.55]
buffer = 0.25
def get_coord(act):
    x_coord = BIN_X_COORD[0] - 0.5 * buffer + act[0] * (BIN_X_COORD[3] - BIN_X_COORD[0] + buffer)
    y_coord = BIN_Y_COORD[0] - 0.5 * buffer + act[1] * (BIN_Y_COORD[6] - BIN_Y_COORD[0] + buffer)
    z_coord = act[2] *  BIN_Z_COORD[0]
    return x_coord, y_coord, z_coord

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(env, hidden_sizes=32, lr=1e-2, epochs=50, batch_size=1000, substep=250):
    
    # make environment, check spaces, get obs / act dims
    # env = gym.make(env_name)
    # assert isinstance(env.observation_space, Dict), \
    #     "This example only works for envs with continuous state spaces."
    # assert isinstance(env.action_space, Box), \
    #     "This example only works for envs with discrete action spaces."
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.shape[0]
    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+[hidden_sizes]+[n_acts])

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return  ContinuousBernoulli(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        act = get_policy(obs).sample()
        print("original", act)
        act = torch.as_tensor(get_coord(act))
        print("coord", act)
        return act

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:
            
            try:
                # # rendering
                # if (not finished_rendering_this_epoch) and render:
                #     env.render()
                # save obs
                batch_obs.append(copy.deepcopy(obs[0]))

                # act in the environment
                act = get_action(torch.as_tensor(obs[0]))
                
                for i in range(substep):
                    obs, rew, done, _ = env.step(act)
                print('Episode', len(ep_rews) + 1)
                print('obs', obs[0])
                print('rew', rew[0])
                print('done', done)
                print('_', _)
                # save action, reward
                batch_acts.append(act)
                ep_rews.append(rew[0])

                if done:
                    # if episode is over, record info about episode
                    ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                    batch_rets.append(ep_ret)
                    batch_lens.append(ep_len)

                    # the weight for each logprob(a|s) is R(tau)
                    batch_weights += [ep_ret] * ep_len

                    # reset episode-specific variables
                    obs, done, ep_rews = env.reset(), False, []

                    # won't render again this epoch
                    finished_rendering_this_epoch = True

                    # end experience loop if we have enough of it
                    if len(batch_obs) > batch_size:
                        break
            except KeyboardInterrupt:
                print('Closing')
                done = True
                break

        env.close()
        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))


# if __name__ == '__main__':

#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--env_name', '--env', type=str, default='arm_training_env_pg-v0')
#     parser.add_argument('--render', action='store_true')
#     parser.add_argument('--lr', type=float, default=1e-2)
#     args = parser.parse_args()
#     print('\nUsing simplest formulation of policy gradient.\n')
#     train(env_name=args.env_name, lr=args.lr)