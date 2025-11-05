import os
from stable_baselines3.common.monitor import Monitor
from sympy import false

from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from residual_hopper import ResidualHopperEnv, ObsWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor


def main():
    te=50
    env = gym.make('CustomHopper-target-v0')
    base_policy = PPO.load("PPOSourceUDR.zip", env=env)
    env = ResidualHopperEnv(env, base_policy)
    #env = ObsWrapper(env)
    #env.modify_obstacles(0,0)
    log_dir = "./tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    name = "last.zip"
    env = VecNormalize.load(name +"_vecnormalize.pkl", env)
    env.training=False
    env.norm_reward=False
    model=PPO.load(name, env=env)
    mean_reward, std_reward = evaluate_policy(model, env, te, render=False)
    print(f"Test reward (avg +/- std): ({mean_reward} +/- {std_reward}) - Num episodes: {te}")
if __name__ == '__main__':
    main()