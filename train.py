"""Sample script for training a control policy on the Hopper environment

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between TRPO, PPO, and SAC.
"""
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from sympy import false

from env.custom_hopper import *
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
import torch
from residual_hopper import ResidualHopperEnv, ObsWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

class ResetWeightsCallback(BaseCallback):
    def __init__(self, reset_function, verbose=0):
        super(ResetWeightsCallback, self).__init__(verbose)
        self.reset_function = reset_function
        self.episode_count = 0
        self.executed = False
    def _on_step(self) -> bool:

        if self.locals["dones"].any():
            self.episode_count += 1
            critic_loss = self.model.logger.name_to_value.get("train/value_loss", None)
            if critic_loss is not None and critic_loss < 200 and not self.executed:
                self.executed = True
                with torch.no_grad():
                    self.model.policy.log_std.copy_(torch.zeros_like(self.model.policy.log_std))
                    print(self.model.policy.log_std)
                for name, param in self.model.policy.named_parameters():
                    if 'action_net' in name:
                        param.requires_grad = True
        return True

def initialize_policy(policy):
    with torch.no_grad():
        policy.action_net.weight.fill_(0.0)
        policy.action_net.bias.fill_(0.0)
        policy.log_std.fill_(-10)

    for name, param in policy.named_parameters():
        if 'action_net' in name:
            param.requires_grad = False

def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

def plot_results(log_folder, title="Learning Curve"):
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    x = x[len(x) - len(y) :]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()

def main():


    env = gym.make('CustomHopper-source-v0')
    base_policy=PPO.load("PPOSourceUDR.zip", env=env)
    env=ResidualHopperEnv(env, base_policy)
    #env=ObsWrapper(env)
    #env.modify_obstacles(0, 0)
    lr=0.0003
    seed=0
    ts=1000000
    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper
    log_dir = "./tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    env=DummyVecEnv([lambda: env])
    env=VecNormalize(env,norm_obs=True, norm_reward=False)
    #model = PPO.load("EXT_PRLUDR.zip", env=env)
    model =PPO("MlpPolicy", env,learning_rate=lr, seed=seed, verbose=1)
    initialize_policy(model.policy)
    reset_callback = ResetWeightsCallback(initialize_policy, verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=200000, save_path='./model_checkpoints/')
    model.learn(total_timesteps=ts, callback=[reset_callback,checkpoint_callback])
    name="last.zip"
    model.save(name)
    env.save(name+"_vecnormalize.pkl")
    plot_results(log_dir)
    print("Saved model")
    """
        TODO:
            - train a policy with stable-baselines3 on the source Hopper env 
            - test the policy with stable-baselines3 on <source,target> Hopper envs (hint: see the evaluate_policy method of stable-baselines3)
    """

if __name__ == '__main__':
    main()