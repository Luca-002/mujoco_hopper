import numpy as np
from gym import Wrapper
from gym.spaces import Box
class ResidualHopperEnv(Wrapper):
    def __init__(self, env, base_policy):
        super().__init__(env)
        self.base_policy = base_policy

        observation= env.reset()
        self.last_obs=observation
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        self.action_space = Box(
            low=env.action_space.low * 2,
            high=env.action_space.high * 2,
            dtype=np.float32
        )

    def step(self, residual_action):
        base_action, _= self.base_policy.predict(self.last_obs, deterministic=True)
        combined_action = np.clip(
            base_action + residual_action,
            self.action_space.low,
            self.action_space.high
        )
        obs, reward, done, info = self.env.step(combined_action)
        self.last_obs=obs
        obs = np.concatenate((obs, self.env.get_obstacles()))
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.last_obs = self.env.reset(**kwargs)
        obs = np.concatenate((self.last_obs, self.env.get_obstacles()))
        return obs
class ObsWrapper(Wrapper):
    def __init__(self, env, ):
        super().__init__(env)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.concatenate((obs, self.env.get_obstacles()))
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = np.concatenate((obs, self.env.get_obstacles()))
        return obs