import gymnasium as gym
import numpy as np
from gymnasium import spaces

from BA_Thesis_Project.SafetyEnvelope import SafetyEnvelope
from BA_Thesis_Project.SafetyEnvelopeWrapper import SafetyEnvelopeWrapper
from gymnasium_envCustomHue.envs import FrontalEncounter

SAFETY_ENVELOPE_INTERVENE_PENALTY_SCALE= -80

class SafetyEnvelopeWrapperB1(gym.Wrapper):
    def __init__(self, env: gym.Env, safety_envelope_intervene_penalty_scale: float = SAFETY_ENVELOPE_INTERVENE_PENALTY_SCALE):
        """
            :param env: the base env
            :param safety_envelope_intervene_penalty_scaler: penalty scaler when Safety Envelope intervenes
        """
        super().__init__(env)
        self.safety_envelope_intervenes = False
        self.alertness_value = 0.0
        self.alertnessHuman = 0.0
        self.alertnessWall = 0.0
        # safe penalty scaler to re-compute the reward later in function step()
        self.safety_envelope_intervene_penalty_scale= safety_envelope_intervene_penalty_scale
        # save current obs, info, observed by the Safety Envelope Wrapper
        self.current_obs = None
        self.current_info = None
        # helper class to do the actual Safety Envelope logic
        self.safetyEnvelopeImp = SafetyEnvelope(env)
        # Start from the underlying Dict space
        # ValueError: cannot reshape array of size 0 into shape (0,) get when run EvalCallBack. Cause:SB3 expects every observation key to have a fixed, non-zero shape.But SocNavGym returned empty arrays for:
        # plants: shape (0,),.../SB3 then tried to reshape these during evaluation:obs_[key].reshape((-1, *self.observation_space[key].shape))reshaping size-0 into shape (0,) crashes.
        #Even if your config sets min/max = 0, SocNavGym still registers these keys in observation_space, so SB3 thinks they always exist, but the returned obs is empty → mismatch.
        #-> need to pruned spaces of obs. remove all key that do not appear in env
        pruned_spaces = {}
        original_space= self.env.observation_space
        assert isinstance(original_space, spaces.Dict), "SocNavGym obs must be Dict"
        pruned_spaces["robot"]= original_space["robot"]
        # for humans: keep only the first 11 fields per entity -> change the shape of humans
        humans_box= original_space["humans"]
        assert isinstance(humans_box, spaces.Box)
        low_original= humans_box.low
        high_original= humans_box.high
        # reshape to (num_entities, 14), crop to first 11, flatten back
        low_cropped= low_original.reshape(-1,14)[:, :11].flatten()
        high_cropped = high_original.reshape(-1, 14)[:, :11].flatten()
        pruned_spaces["humans"]=spaces.Box(low=low_cropped, high=high_cropped, shape=low_cropped.shape, dtype=humans_box.dtype)
        # # for walls
        # if "walls" in original_space.spaces:
        #     walls_box=original_space["walls"]
        #     assert isinstance(walls_box, spaces.Box)
        #     low_walls_original=walls_box.low
        #     high_walls_original=walls_box.high
        #     low_walls_cropped = low_walls_original.reshape(-1, 14)[:, :11].flatten()
        #     high_walls_cropped = high_walls_original.reshape(-1, 14)[:, :11].flatten()
        #     pruned_spaces["walls"]=spaces.Box(low=low_walls_cropped, high=high_walls_cropped, shape=low_walls_cropped.shape, dtype=walls_box.dtype)

        self.observation_space = spaces.Dict(pruned_spaces)



    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # reset own wrapper's attributes, which is not in base env
        self.safety_envelope_intervenes = False
        self.alertness_value = 0.0
        self.alertnessHuman = 0.0
        self.alertnessWall = 0.0
        # modified based env's info -> wrapper's info
        # create a safe copy of original dict obs, info so that the wrapper can add new field but the underlying base env stay untouched.
        info = dict(info)
        info["safety_envelope_intervenes"] = self.safety_envelope_intervenes

        obs = dict(obs)
        # drop unused entities:only keep keys in pruned obs_space ("robot", "humans")
        obs=self.filter_obs(obs)
        # filter entity observations: drop indices 11–13

        for key in ["humans", "plants", "laptops", "tables", "walls"]:
            if key in obs:
                obs[key] = SafetyEnvelopeWrapperB1.filter_unrelated_field_value(obs[key])

        observation_robot=obs

        # # IMPORTANT: need to run reset before doing anything else to set the initial minimum distance to other obstacles in info to use later in step()
        # initial_minimum_distance_Obstacles = self.compute_initial_d_min_to_obstacles(obs)
        # info['MINIMUM_OBSTACLE_DISTANCE'] = initial_minimum_distance_Obstacles

        self.current_obs = obs
        self.current_info = info
        return self.current_obs, self.current_info

    def step(self, action):
        print("...................................................................................................................")
        safe_action,  alertness_value, safety_envelope_intervenes,alertHuman, alertWall = self.safetyEnvelopeImp.next_action(  action, self.current_obs)
        self.alertness_value = alertness_value
        self.alertnessHuman=alertHuman
        self.alertnessWall= alertWall

        self.safety_envelope_intervenes = safety_envelope_intervenes
        # now step the base env with the safe action
        obs, reward, terminated, truncated, info = self.env.step(safe_action)

        # modified based env's obs, info -> wrapper's obs, info
        obs = dict(obs)
        info = dict(info)
        obs=self.filter_obs(obs)
        # filter entity observations: drop indices 11–13
        obs = dict(obs)
        for key in ["humans", "plants", "laptops", "tables", "walls"]:
            if key in obs:
                obs[key] = SafetyEnvelopeWrapperB1.filter_unrelated_field_value(obs[key])

        info["safety_envelope_intervenes"] = self.safety_envelope_intervenes
        self.current_obs = obs
        self.current_info = info
        # recompute the reward from the base reward
        if self.safety_envelope_intervenes:
            reward += self.safety_envelope_intervene_penalty_scale
            terminated = True
        # set terminated to True when the safety envelope need to intervene to end the episode
        return self.current_obs, reward, terminated, truncated, self.current_info


    @staticmethod
    def filter_unrelated_field_value(array: np.ndarray) -> np.ndarray:
        if array.size == 0 or array is None:
            return array
        # split 1-D array into rows, one per entity ->shape(-1,14) :[]-> [[],..[]]
        entities = array.reshape(-1, 14)
        # remove unrelated field value, remove relative speed and gaze from obs
        filtered_array = entities[:, :11]

        return filtered_array.flatten().astype(np.float32)

    def filter_obs(self, obs: dict):
        """Return only the keys allowed by pruned observation_space.here: keys in wrapper obs space is robot and humans, defined in init()"""
        allowed = self.observation_space.spaces.keys()
        print(f"alloed key {allowed}")
        return {k: obs[k] for k in allowed if k in obs}
