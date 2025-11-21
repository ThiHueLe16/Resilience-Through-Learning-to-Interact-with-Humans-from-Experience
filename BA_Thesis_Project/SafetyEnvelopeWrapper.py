import gymnasium as gym
import numpy as np
from gymnasium import spaces

from BA_Thesis_Project.SafetyEnvelope import SafetyEnvelope

ALERTNESS_VALUE_PENALTY_SCALE= 1000
SAFETY_ENVELOPE_INTERVENE_PENALTY_SCALE= -20000


class SafetyEnvelopeWrapper(gym.Wrapper):
    def __init__(self, env:gym.Env, expose_alertness_value_in_obs:bool=False, shape_reward_with_alertness_value:bool=False, alertness_penalty_scale:float=ALERTNESS_VALUE_PENALTY_SCALE,
                 safety_envelope_intervene_penalty_scaler:float=SAFETY_ENVELOPE_INTERVENE_PENALTY_SCALE):
        """

        :param env: the base env
        :param expose_alertness_value_in_obs: if True, add the alertness value to the obs of the wrapper. Use for proposed Resilience Architecture
        :param shape_reward_with_alertness_value:  if True, reward was shaped using alertness value as well (used during training for the proposed Resilience Architecture).
        :param alertness_penalty_scale:
        """
        super().__init__(env)
        self.expose_alertness_value_in_obs=expose_alertness_value_in_obs
        self.shape_reward_with_alertness_value=shape_reward_with_alertness_value
        self.alertness_penalty_scale= alertness_penalty_scale

        self.alertness_value=0.0
        self.safety_envelope_intervenes=False
        # save current obs, info, observed by the Safety Envelope Wrapper
        self.current_obs= None
        self.current_info=None
        # helper class to do the actual Safety Envelope logic
        self.safetyEnvelopeImp= SafetyEnvelope(env)
        # safe penalty scaler to re-compute the reward later in function step()
        self.alertness_penalty_scale= alertness_penalty_scale
        self.safety_envelope_intervene_penalty_scaler= safety_envelope_intervene_penalty_scaler

        # update the observation space of the wrapper
        original_space= env.observation_space
        if self.expose_alertness_value_in_obs:
            # add alertness_value to the obs of wrapper sothat the RL agent can learn from observing obs and reward
            assert isinstance(original_space, spaces.Dict),"SocNavGym obs must be Dict"
            self.observation_space=spaces.Dict({**original_space,"alertness_value": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),})
        else:
            self.observation_space=original_space

    def reset(self, **kwargs):
        obs, info= self.env.reset(**kwargs)
        # reset own wrapper's attributes, which is not in base env
        self.alertness_value = 0.0
        self.safety_envelope_intervenes= False

        # modified based env's obs, info -> wrapper's obs, info
        # create a safe copy of original dict obs, info so that the wrapper can add new field but the underlying base env stay untouched.
        obs = dict(obs)
        info= dict(info)
        if self.expose_alertness_value_in_obs:
            obs["alertness_value"]=np.array([self.alertness_value],dtype=np.float32)
            info["alertness_value"] = self.alertness_value
        info["safety_envelope_intervenes"] = self.safety_envelope_intervenes
        # obs["safety_envelope_intervenes"]= self.safety_envelope_intervenes
        self.current_obs = obs
        self.current_info = info

        return self.current_obs, self.current_info

    def step(self, action):
        min_distance_to_obstacles= self.current_info['MINIMUM_OBSTACLE_DISTANCE']
        safe_action, alertness_value, safety_envelope_intervenes= self.safetyEnvelopeImp.next_action(min_distance_to_obstacles, action, self.current_obs)

        self.alertness_value= alertness_value
        self.safety_envelope_intervenes= safety_envelope_intervenes
        # now step the base env with the safe action
        obs, reward, terminated, truncated, info= self.env.step(safe_action)

        # modified based env's obs, info -> wrapper's obs, info
        obs = dict(obs)
        info=dict(info)
        if self.expose_alertness_value_in_obs:
            #  add alertness_value to obs for proposed resilience Architecture
            obs["alertness_value"]=np.array([self.alertness_value],dtype=np.float32)
            info["alertness_value"] = self.alertness_value
        info["safety_envelope_intervenes"] = self.safety_envelope_intervenes
        # # TODO: CHECK IF should add safety envelopwe intervene inside the obs
        # obs["safety_envelope_intervenes"] = self.safety_envelope_intervenes
        self.current_obs = obs
        self.current_info = info

        # recompute the reward from the base reward
        if self.safety_envelope_intervenes:
            reward -= self.safety_envelope_intervene_penalty_scaler
        # TODO: check if reward shaping by using alertness value should be used by both training and evaluation. I think for both. if yes. remove the shape_reward_with_alertness_value abd expose_alerness_to_obs in init() with only 1 propose_RA:bool=False (or use_alerness_value)
        # for proposed Resilience Architecture during training only
        if self.shape_reward_with_alertness_value:
            reward -= self.alertness_penalty_scale*self.alertness_value
        # set terminated to True when the safety envelope need to intervene to end the episode
        if self.safety_envelope_intervenes:
            terminated=True

        return self.current_obs, reward, terminated, truncated, self.current_info


















