import gymnasium as gym
import numpy as np
from gymnasium import spaces

from BA_Thesis_Project.SafetyEnvelope import SafetyEnvelope

SAFETY_ENVELOPE_INTERVENE_PENALTY_SCALE= -20000

class SafetyEnvelopeWrapperB1(gym.Wrapper):
    def __init__(self, env: gym.Env, safety_envelope_intervene_penalty_scale: float = SAFETY_ENVELOPE_INTERVENE_PENALTY_SCALE):
        """
            :param env: the base env
            :param safety_envelope_intervene_penalty_scaler: penalty scaler when Safety Envelope intervenes
        """
        super().__init__(env)
        self.safety_envelope_intervenes = False
        self.alertness_value = 0.0
        # safe penalty scaler to re-compute the reward later in function step()
        self.safety_envelope_intervene_penalty_scale= safety_envelope_intervene_penalty_scale
        # save current obs, info, observed by the Safety Envelope Wrapper
        self.current_obs = None
        self.current_info = None
        # helper class to do the actual Safety Envelope logic
        self.safetyEnvelopeImp = SafetyEnvelope(env)


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # reset own wrapper's attributes, which is not in base env
        self.safety_envelope_intervenes = False
        self.alertness_value = 0.0
        # modified based env's info -> wrapper's info
        # create a safe copy of original dict obs, info so that the wrapper can add new field but the underlying base env stay untouched.
        info = dict(info)
        info["safety_envelope_intervenes"] = self.safety_envelope_intervenes
        self.current_obs = obs
        self.current_info = info
        return self.current_obs, self.current_info

    def step(self, action):
        min_distance_to_obstacles = self.current_info['MINIMUM_OBSTACLE_DISTANCE']
        safe_action,  alertness_value, safety_envelope_intervenes = self.safetyEnvelopeImp.next_action( min_distance_to_obstacles, action, self.current_obs)
        self.alertness_value = alertness_value
        self.safety_envelope_intervenes = safety_envelope_intervenes
        # now step the base env with the safe action
        obs, reward, terminated, truncated, info = self.env.step(safe_action)

        # modified based env's obs, info -> wrapper's obs, info
        obs = dict(obs)
        info = dict(info)

        info["safety_envelope_intervenes"] = self.safety_envelope_intervenes
        self.current_obs = obs
        self.current_info = info
        # recompute the reward from the base reward
        if self.safety_envelope_intervenes:
            reward -= self.safety_envelope_intervene_penalty_scale
            terminated = True
        # set terminated to True when the safety envelope need to intervene to end the episode
        return self.current_obs, reward, terminated, truncated, self.current_info
