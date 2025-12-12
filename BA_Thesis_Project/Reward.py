from typing import Any

import numpy as np
from pandas.core.indexers import unpack_1tuple
from socnavgym.envs.rewards import RewardAPI
from socnavgym.envs.socnavenv_v1 import SocNavEnv_v1
from socnavgym.envs.socnavenv_v1 import EntityObs

import sys, os, importlib, inspect

# repo_path = "/Users/thihuele/HumanDetection/navgym/SocNavGym"
# if repo_path not in sys.path:
#     sys.path.insert(0, repo_path)
#
# # # If socnavgym was already imported from site-packages, purge it
# if 'socnavgym' in sys.modules:
#     del sys.modules['socnavgym']

import socnavgym
print("Using socnavgym from:", inspect.getfile(socnavgym))

class Reward(RewardAPI):
    def __init__(self, env:Any):
        super().__init__(env)

        # self.env.alertness_value=None
        # self.env.safety_envelope_intervenes= False
        self.prev_distance= None
        self.distancePenalty=-3.0
        self.reach_reward = 30
        self.collision_penalty = -50
        self.max_steps_penalty= -10
        self.step_penalty=-0.02

    def compute_reward(self, action, prev_obs: EntityObs, curr_obs: EntityObs):
        """
            Compute the base reward after running an action.
        """

        if self.check_reached_goal():
            return self.reach_reward

        elif self.check_collision():
            # this case should not happen for the baseline architecture and the proposed architecture because the safety envelope need to intervene and swap action before any collision happen
            # just add here to fulfil the logic or incase we want to test the RL agent without safety envelope shield as well.
            return self.collision_penalty
        elif self.check_timeout(): return self.max_steps_penalty
        else:
            distance_to_goal = np.sqrt((self.env.robot.goal_x - self.env.robot.x) ** 2 + (self.env.robot.goal_y - self.env.robot.y) ** 2)

            reward=0

            if self.prev_distance is not None:
                reward = self.step_penalty+(distance_to_goal-self.prev_distance) *self.distancePenalty

            self.prev_distance = distance_to_goal
            return reward



