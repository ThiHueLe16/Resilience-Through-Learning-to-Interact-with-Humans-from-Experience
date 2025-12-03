# import gymnasium as gym
# import numpy as np
#
# from gymnasium import spaces
#
# from BA_Thesis_Project.SafetyEnvelope import SafetyEnvelope
# from BA_Thesis_Project.SafetyEnvelopeWrapper import SafetyEnvelopeWrapper
# from gymnasium_envCustomHue.envs import FrontalEncounter
# from testRemoveKeyObs import space
#
# SAFETY_ENVELOPE_INTERVENE_PENALTY_SCALE= -20000
#
# class SafetyEnvelopeWrapperB1(gym.Wrapper):
#     def __init__(self, env: gym.Env, safety_envelope_intervene_penalty_scale: float = SAFETY_ENVELOPE_INTERVENE_PENALTY_SCALE):
#         """
#             :param env: the base env
#             :param safety_envelope_intervene_penalty_scaler: penalty scaler when Safety Envelope intervenes
#         """
#         super().__init__(env)
#         self.safety_envelope_intervenes = False
#         self.alertness_value = 0.0
#         # safe penalty scaler to re-compute the reward later in function step()
#         self.safety_envelope_intervene_penalty_scale= safety_envelope_intervene_penalty_scale
#         # save current obs, info, observed by the Safety Envelope Wrapper
#         self.current_obs = None
#         self.current_info = None
#         # helper class to do the actual Safety Envelope logic
#         self.safetyEnvelopeImp = SafetyEnvelope(env)
#         # Start from the underlying Dict space
#         # ValueError: cannot reshape array of size 0 into shape (0,) get when run EvalCallBack. Cause:SB3 expects every observation key to have a fixed, non-zero shape.But SocNavGym returned empty arrays for:
#         # plants: shape (0,),.../SB3 then tried to reshape these during evaluation:obs_[key].reshape((-1, *self.observation_space[key].shape))reshaping size-0 into shape (0,) crashes.
#         #Even if your config sets min/max = 0, SocNavGym still registers these keys in observation_space, so SB3 thinks they always exist, but the returned obs is empty → mismatch.
#         #-> need to pruned spaces of obs. remove all key that do not appear in env
#         pruned_spaces = {}
#         original_space= self.env.observation_space
#         assert isinstance(original_space, spaces.Dict), "SocNavGym obs must be Dict"
#         # Keep only the useful, non-empty keys
#         # fix error: can not reshape array of size 0 into shape(0,) happen when run evaluation by SB3.-> remove all entity like plant, table... which has shape (0,)
#         # for key, space in original_space.spaces.items():
#         #     # Drop zero-length boxes completely
#         #     if isinstance(space, spaces.Box) and space.shape == (0,):
#         #         continue
#         #     # Keep robot and humans (and anything else non-empty you care about)
#         #     if key in ("robot", "humans"):
#         #         pruned_spaces[key] = space
#         #         # Replace this wrapper's observation_space with the pruned Dict
#         pruned_spaces["robot"]= original_space["robot"]
#         pruned_spaces["walls"]=original_space["walls"]
#         # for humans: keep only the first 11 fields per entity -> change the shape of humans
#         humans_box= original_space["humans"]
#         assert isinstance(humans_box, spaces.Box)
#         low_original= humans_box.low
#         high_original= humans_box.high
#         # reshape to (num_entities, 14), crop to first 11, flatten back
#         low_cropped= low_original.reshape(-1,14)[:, :11].flatten()
#         high_cropped = high_original.reshape(-1, 14)[:, :11].flatten()
#         pruned_spaces["humans"]=spaces.Box(low=low_cropped, high=high_cropped, shape=low_cropped.shape, dtype=humans_box.dtype)
#         self.observation_space = spaces.Dict(pruned_spaces)
#
#
#
#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         # reset own wrapper's attributes, which is not in base env
#         self.safety_envelope_intervenes = False
#         self.alertness_value = 0.0
#         # modified based env's info -> wrapper's info
#         # create a safe copy of original dict obs, info so that the wrapper can add new field but the underlying base env stay untouched.
#         info = dict(info)
#         info["safety_envelope_intervenes"] = self.safety_envelope_intervenes
#
#         obs = dict(obs)
#         # drop unused entities:only keep keys in pruned obs_space ("robot", "humans")
#         obs=self.filter_obs(obs)
#         # filter entity observations: drop indices 11–13
#
#         for key in ["humans", "plants", "laptops", "tables", "walls"]:
#             if key in obs:
#                 obs[key] = SafetyEnvelopeWrapperB1.filter_unrelated_field_value(obs[key])
#
#         # IMPORTANT: need to run reset before doing anything else to set the initial minimum distance to other obstacles in info to use later in step()
#         initial_minimum_distance_Obstacles = self.compute_initial_d_min_to_obstacles(obs)
#         info['MINIMUM_OBSTACLE_DISTANCE'] = initial_minimum_distance_Obstacles
#
#         self.current_obs = obs
#         self.current_info = info
#         return self.current_obs, self.current_info
#
#     def step(self, action):
#         min_distance_to_obstacles = self.current_info['MINIMUM_OBSTACLE_DISTANCE']
#         safe_action,  alertness_value, safety_envelope_intervenes = self.safetyEnvelopeImp.next_action( min_distance_to_obstacles, action, self.current_obs)
#         self.alertness_value = alertness_value
#         self.safety_envelope_intervenes = safety_envelope_intervenes
#         # now step the base env with the safe action
#         obs, reward, terminated, truncated, info = self.env.step(safe_action)
#
#         # modified based env's obs, info -> wrapper's obs, info
#         obs = dict(obs)
#         info = dict(info)
#         obs=self.filter_obs(obs)
#         # filter entity observations: drop indices 11–13
#         obs = dict(obs)
#         for key in ["humans", "plants", "laptops", "tables", "walls"]:
#             if key in obs:
#                 obs[key] = SafetyEnvelopeWrapperB1.filter_unrelated_field_value(obs[key])
#
#         info["safety_envelope_intervenes"] = self.safety_envelope_intervenes
#         self.current_obs = obs
#         self.current_info = info
#         # recompute the reward from the base reward
#         if self.safety_envelope_intervenes:
#             reward += self.safety_envelope_intervene_penalty_scale
#             terminated = True
#         # set terminated to True when the safety envelope need to intervene to end the episode
#         return self.current_obs, reward, terminated, truncated, self.current_info
#
#     @staticmethod
#     def compute_initial_d_min_to_obstacles(obs):
#         """
#             Return the initial minimum distance to all others obstacles. Need only for the begining, after 1 first step,
#             this distance is returned in the info after executing env.step(action)
#         """
#         d_min = float('inf')
#         # extract robot radius from the observation
#         robot_radius = obs["robot"][8]
#
#         # predict new distance from robot to each obstacle and find the minimum
#         for key in ["humans", "plants", "laptops", "tables", "walls"]:
#             if key not in obs:
#                 continue
#             obstacle_obs = obs[key]
#             if obstacle_obs.size == 0:
#                 continue
#             assert len(obstacle_obs) % 11 == 0
#             number_of_obstacles = int(len(obstacle_obs) / 11)
#             for i in range(0, number_of_obstacles):
#                 obstacle_relative_position_x = obstacle_obs[6 + i * 11]
#                 obstacle_relative_position_y = obstacle_obs[7 + i * 11]
#                 obstacle_radius = obstacle_obs[10 + i * 11]
#                 distance= np.sqrt(obstacle_relative_position_x**2 + obstacle_relative_position_y**2 ) -robot_radius -obstacle_radius
#                 if d_min > distance:import gymnasium as gym

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from BA_Thesis_Project.SafetyEnvelope import SafetyEnvelope
from BA_Thesis_Project.SafetyEnvelopeWrapper import SafetyEnvelopeWrapper
from gymnasium_envCustomHue.envs import FrontalEncounter

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
        # Start from the underlying Dict space
        # ValueError: cannot reshape array of size 0 into shape (0,) get when run EvalCallBack. Cause:SB3 expects every observation key to have a fixed, non-zero shape.But SocNavGym returned empty arrays for:
        # plants: shape (0,),.../SB3 then tried to reshape these during evaluation:obs_[key].reshape((-1, *self.observation_space[key].shape))reshaping size-0 into shape (0,) crashes.
        #Even if your config sets min/max = 0, SocNavGym still registers these keys in observation_space, so SB3 thinks they always exist, but the returned obs is empty → mismatch.
        #-> need to pruned spaces of obs. remove all key that do not appear in env
        pruned_spaces = {}
        original_space= self.env.observation_space
        print(f"original observation space ={original_space.spaces.keys()}")
        assert isinstance(original_space, spaces.Dict), "SocNavGym obs must be Dict"
        # Keep only the useful, non-empty keys
        # fix error: can not reshape array of size 0 into shape(0,) happen when run evaluation by SB3.-> remove all entity like plant, table... which has shape (0,)
        # for key, space in original_space.spaces.items():
        #     # Drop zero-length boxes completely
        #     if isinstance(space, spaces.Box) and space.shape == (0,):
        #         continue
        #     # Keep robot and humans (and anything else non-empty you care about)
        #     if key in ("robot", "humans"):
        #         pruned_spaces[key] = space
        #         # Replace this wrapper's observation_space with the pruned Dict
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
        self.observation_space = spaces.Dict(pruned_spaces)



    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # reset own wrapper's attributes, which is not in base env
        self.safety_envelope_intervenes = False
        self.alertness_value = 0.0
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
        safe_action,  alertness_value, safety_envelope_intervenes = self.safetyEnvelopeImp.next_action(  action, self.current_obs)
        self.alertness_value = alertness_value
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

    # def compute_min_distance_to_wall(self,obs):
    #     # fix error key walls do not exist in obs because of setting padding to True to prevent failure of reshape from SB3
    #     robot_radius = obs["robot"][8]
    #     robot_x_world_frame = self.env.get_wrapper_attr("robot").x
    #     robot_y_world_frame = self.env.get_wrapper_attr("robot").y
    #     MAP_Y = self.env.get_wrapper_attr("MAP_Y")
    #     MARGIN = self.env.get_wrapper_attr("MARGIN")
    #     MAP_X = self.env.get_wrapper_attr("MAP_X")
    #     HALF_SIZE_X = MAP_X / 2 - MARGIN
    #     HALF_SIZE_Y = MAP_Y / 2 - MARGIN
    #
    #     # Distance to each wall (subtract robot radius)
    #     d_left = (robot_x_world_frame - (-HALF_SIZE_X)) - robot_radius  # distance to left wall
    #     d_right = (HALF_SIZE_X - robot_x_world_frame) - robot_radius  # distance to right wall
    #     d_bottom = (robot_y_world_frame - (-HALF_SIZE_Y)) - robot_radius  # distance to bottom wall
    #     d_top = (HALF_SIZE_Y - robot_y_world_frame) - robot_radius  # distance to top wall
    #     print(f"dtop {d_top}")
    #     print(f"dbottom {d_bottom}")
    #     print(f"dleft {d_left}")
    #     print(f"dright {d_right}")
    #     d_min=min(d_left, d_right, d_bottom, d_top)
    #     print(f"HELLO GET min dis to wall {d_min}")
    #
    #     # The minimum distance to any wall
    #     return d_min
    #
    #
    # @staticmethod
    # def compute_initial_d_min_to_obstacles(self, obs):
    #     """
    #         Return the initial minimum distance to all others obstacles (NOT TO WALLS, bc of setting padding to prevent error of SB3). Need only for the begining, after 1 first step,
    #         this distance is returned in the info after executing env.step(action)
    #
    #     """
    #     d_min = float('inf')
    #     # extract robot radius from the observation
    #     robot_radius = obs["robot"][8]
    #
    #     # predict new distance from robot to each obstacle and find the minimum, walls is not in keys bc of setting padding to True(prevent error from SB3)
    #     # for key in ["humans", "plants", "laptops", "tables", "walls"]:
    #     for key in ["humans", "plants", "laptops", "tables"]:
    #         if key not in obs:
    #             continue
    #         obstacle_obs = obs[key]
    #         if obstacle_obs.size == 0:
    #             continue
    #         assert len(obstacle_obs) % 11 == 0
    #         number_of_obstacles = int(len(obstacle_obs) / 11)
    #         for i in range(0, number_of_obstacles):
    #             obstacle_relative_position_x = obstacle_obs[6 + i * 11]
    #             obstacle_relative_position_y = obstacle_obs[7 + i * 11]
    #             obstacle_radius = obstacle_obs[10 + i * 11]
    #             distance= np.sqrt(obstacle_relative_position_x**2 + obstacle_relative_position_y**2 ) -robot_radius -obstacle_radius
    #             if d_min > distance:
    #                 d_min =distance
    #     return d_min



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
