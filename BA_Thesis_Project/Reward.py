from typing import Any

import numpy as np
from pandas.core.indexers import unpack_1tuple
from socnavgym.envs.rewards import RewardAPI
from socnavgym.envs.socnavenv_v1 import SocNavEnv_v1
from socnavgym.envs.socnavenv_v1 import EntityObs

import sys, os, importlib, inspect

repo_path = "/Users/thihuele/HumanDetection/navgym/SocNavGym"
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

# # If socnavgym was already imported from site-packages, purge it
if 'socnavgym' in sys.modules:
    del sys.modules['socnavgym']

import socnavgym
print("Using socnavgym from:", inspect.getfile(socnavgym))

class Reward(RewardAPI):
    def __init__(self, env:Any):
        super().__init__(env)

        # self.env.alertness_value=None
        # self.env.safety_envelope_intervenes= False
        self.prev_distance= None

        self.distance_reward_scaler=50
        self.alertness_value_scaler=50
        self.reach_reward = 20000
        self.intervention_reward=-20000


    # def set_alertness_value(self, updated_value):
    #     """
    #         In the proposed RA, the Safety Envelope does not only check for intervention but also output an
    #         additional alertness value, which computes the probability to intervention threshold.
    #         This alertness value is set via this func before calling compute_reward(). compute_reward() need this value to calculate the reward
    #     """
    #     self.alertness_value= updated_value
    #     return
    #
    # def set_safety_envelope_intervenes(self, intervene_boolean):
    #     """
    #         set safety_envelope_intervenes with the given intervene_boolean, which is True if Safety Envelope need to intervene and False otherwise
    #     """
    #     self.safety_envelope_intervenes= intervene_boolean



    def compute_reward(self, action, prev_obs: EntityObs, curr_obs: EntityObs):
        """
            Compute the reward after running an action.

        """
        print(f" hello HUEEEEEEE dang check gia tri safety signal {self.env.alertness_value}")
        if self.check_reached_goal():
            self.info["safety_envelope_intervention"] = False
            return self.reach_reward
        elif self.env.safety_envelope_intervenes:
            self.info["safety_envelope_intervention"] = True
            return self.intervention_reward
        else:
            distance_to_goal = np.sqrt((self.env.robot.goal_x - self.env.robot.x) ** 2 + (self.env.robot.goal_y - self.env.robot.y) ** 2)

            reward=0
            if self.prev_distance is not None:
                reward = -(distance_to_goal-self.prev_distance) * self.distance_reward_scaler

            self.prev_distance = distance_to_goal

            if self.env.alertness_value is not None:
                reward-= self.alertness_value_scaler*self.env.alertness_value
                self.info["alertness_value"] = self.env.alertness_value
            self.info["safety_envelope_intervention"] = False
            return reward

