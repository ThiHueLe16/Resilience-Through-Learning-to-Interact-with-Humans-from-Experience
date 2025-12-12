import numpy as np
import gymnasium as gym
from gymnasium import spaces
from BA_Thesis_Project.SafetyEnvelopeWrapperB2 import SafetyEnvelopeWrapperB2


class SafetyEnvelopeWrapperB3(SafetyEnvelopeWrapperB2):
    def __init__(self, env:gym.Env):
        super().__init__(env)
        # update the observation space of the wrapper
        # original_space = self.env.observation_space
        original_space= self.observation_space
        # add alertness_value to the obs of wrapper sothat the RL agent can learn from observing obs and reward
        assert isinstance(original_space, spaces.Dict), "SocNavGym obs must be Dict"
        self.observation_space = spaces.Dict(
            {**original_space.spaces, "alertness_value": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
             "alertness_human": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
             "alertness_wall": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32), })

    def reset(self, **kwargs):
        obs, info= super().reset( **kwargs)
        info=dict(info)
        # add alertness value to obs, info
        obs["alertness_value"] = np.array([self.alertness_value], dtype=np.float32)
        obs["alertness_human"] = np.array([self.alertnessHuman], dtype=np.float32)
        obs["alertness_wall"] = np.array([self.alertnessWall], dtype=np.float32)
        info["alertness_value"] = self.alertness_value
        info["alertness_human"] = self.alertnessHuman
        info["alertness_wall"] = self.alertnessWall
        obs = dict(obs)
        self.current_obs=obs
        self.current_info=info
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        #  add alertness_value to obs for proposed resilience Architecture
        obs=dict(obs)
        info=dict(info)
        obs["alertness_value"] = np.array([self.alertness_value], dtype=np.float32)
        obs["alertness_human"] = np.array([self.alertnessHuman], dtype=np.float32)
        obs["alertness_wall"] = np.array([self.alertnessWall], dtype=np.float32)

        info["alertness_value"] = self.alertness_value
        info["alertness_human"] = self.alertnessHuman
        info["alertness_wall"] = self.alertnessWall
        self.current_obs=obs
        self.current_info=info
        return self.current_obs, reward, terminated, truncated, self.current_info


