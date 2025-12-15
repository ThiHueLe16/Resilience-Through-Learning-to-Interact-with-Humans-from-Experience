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

class Reward(RewardAPI):
    def __init__(self, env:Any):
        super().__init__(env)
        self.prev_distance= None
        # distancePenalty*progress to give penalty when robot move away from goa;, make progress costly than speed bonus inWrapper 1 to prevent cycling after passing human
        # Tell robot to move toward goal
        self.distancePenalty=-10
        self.reach_reward = 40
        self.collision_penalty = -50
        self.max_steps_penalty= -10
        # encourage robot to finish fast
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
            reward+=self.step_penalty
            if self.prev_distance is None:
                self.prev_distance=distance_to_goal
                return reward

            progress=distance_to_goal-self.prev_distance
            reward += progress*self.distancePenalty

            # penalize being stuck (no progress)
            if abs(progress) < 0.02:
                reward-=0.02
            # update for next step
            self.prev_distance = distance_to_goal
            return reward



