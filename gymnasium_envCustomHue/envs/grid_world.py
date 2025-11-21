import time

import numpy as np
from socnavgym.envs.utils.plant import Plant
from SocNavGym.socnavgym.envs import SocNavEnv_v1
from SocNavGym.socnavgym.envs.socnavenv_v1 import SocNavGymObject
import random
from socnavgym.envs.socnavenv_v1 import SocNavEnv_v1, SocNavGymObject
from socnavgym.envs.utils.plant import Plant


class FrontalEncounter(SocNavEnv_v1):
    def __init__(self, config: str = None, render_mode: str = None) -> None:
        super().__init__(config, render_mode)
        self.alertness_value = None
        self.safety_envelope_intervenes = False

    def _get_kwargs(self, object_type: SocNavGymObject, extra_info: dict = None):
        super()._get_kwargs(object_type, extra_info)
        HALF_SIZE_X = self.MAP_X / 2. - self.MARGIN
        HALF_SIZE_Y = self.MAP_Y / 2. - self.MARGIN
        if object_type == SocNavGymObject.ROBOT:
            arg_dict = {
                "id": 0,  # robot is assigned id 0
                # "x": random.uniform(-HALF_SIZE_X, HALF_SIZE_X),
                # "y": random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                # change this x, y from random to have a fix start on one side-BA THESIS ENV setup
                "x": random.uniform(-HALF_SIZE_X, -HALF_SIZE_X+0.3),
                "y": random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                "theta": random.uniform(-np.pi, np.pi),
                "radius": self.ROBOT_RADIUS,
                "goal_x": None,
                "goal_y": None,
                "goal_a": None,
                "type": self.ROBOT_TYPE
            }
        elif object_type == SocNavGymObject.STATIC_HUMAN or object_type == SocNavGymObject.DYNAMIC_HUMAN:
            policy = self.HUMAN_POLICY
            if policy == "random": policy = random.choice(["sfm", "orca"])
            human_speed = 0
            human_type = "static"
            if object_type == SocNavGymObject.DYNAMIC_HUMAN:
                human_speed = random.uniform(0.0, self.MAX_ADVANCE_HUMAN)
                human_type = "dynamic"
            arg_dict = {
                "id": self.id,
                # "x": random.uniform(HALF_SIZE_X - 5, HALF_SIZE_X),
                # "y": random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                # change (x, y)start position to control the start of human to other side aganst robot-BA THESIS env
                "x": random.uniform(HALF_SIZE_X-0.3, HALF_SIZE_X),
                "y": random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                "theta": random.uniform(-np.pi, np.pi),
                "width": self.HUMAN_DIAMETER,
                "speed": human_speed,
                "goal_radius": self.HUMAN_GOAL_RADIUS,
                "goal_x": None,
                "goal_y": None,
                "policy": policy,
                "fov": self.HUMAN_FOV,
                "prob_to_avoid_robot": self.PROB_TO_AVOID_ROBOT,
                "type": human_type,
                "pos_noise_std": self.HUMAN_POS_NOISE_STD,
                "angle_noise_std": self.HUMAN_ANGLE_NOISE_STD
            }

    def sample_goal(self, goal_radius, object_type: SocNavGymObject, HALF_SIZE_X, HALF_SIZE_Y):
        start_time = time.time()
        while True:
            if self.check_timeout(start_time):
                break
            if object_type == SocNavGymObject.ROBOT:
                goal = Plant(
                    id=None,
                    x=random.uniform(HALF_SIZE_X - 0.3, HALF_SIZE_X),
                    y=random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                    radius=goal_radius
                )
            elif object_type == SocNavGymObject.DYNAMIC_HUMAN:
                goal = Plant(
                    id=None,
                    x=random.uniform(-HALF_SIZE_X, - HALF_SIZE_X + 0.3),
                    y=random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                    radius=goal_radius
                )
            else:
                goal = Plant(
                    id=None,
                    x=random.uniform(-HALF_SIZE_X, HALF_SIZE_X),
                    y=random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                    radius=goal_radius
                )

            collides = False
            all_objects = self.objects
            for obj in (all_objects + list(
                    self.goals.values())):  # check if spawned object collides with any of the exisiting objects. It will not be rendered as a plant.
                if obj is None: continue
                if (goal.collides(obj)):
                    collides = True
                    break

            if collides:
                del goal
            else:
                return goal
        return None

    def try_reset(self, seed=None, options=None):
        super().try_reset(seed, options)
        self.alertness_value = None
        self.safety_envelope_intervenes = False






# from enum import Enum
# import gymnasium as gym
# from gymnasium import spaces
# import pygame
# import numpy as np
#
#
# class Actions(Enum):
#     right = 0
#     up = 1
#     left = 2
#     down = 3
#
#
# class GridWorldEnv(gym.Env):
#     metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
#
#     def __init__(self, render_mode=None, size=5):
#         self.size = size  # The size of the square grid
#         self.window_size = 512  # The size of the PyGame window
#
#         # Observations are dictionaries with the agent's and the target's location.
#         # Each location is encoded as an element of {0, ..., `size`}^2,
#         # i.e. MultiDiscrete([size, size]).
#         self.observation_space = spaces.Dict(
#             {
#                 "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
#                 "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
#             }
#         )
#
#         # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
#         self.action_space = spaces.Discrete(4)
#
#         """
#         The following dictionary maps abstract actions from `self.action_space` to
#         the direction we will walk in if that action is taken.
#         i.e. 0 corresponds to "right", 1 to "up" etc.
#         """
#         self._action_to_direction = {
#             Actions.right.value: np.array([1, 0]),
#             Actions.up.value: np.array([0, 1]),
#             Actions.left.value: np.array([-1, 0]),
#             Actions.down.value: np.array([0, -1]),
#         }
#
#         assert render_mode is None or render_mode in self.metadata["render_modes"]
#         self.render_mode = render_mode
#
#         """
#         If human-rendering is used, `self.window` will be a reference
#         to the window that we draw to. `self.clock` will be a clock that is used
#         to ensure that the environment is rendered at the correct framerate in
#         human-mode. They will remain `None` until human-mode is used for the
#         first time.
#         """
#         self.window = None
#         self.clock = None
#
#     def _get_obs(self):
#         return {"agent": self._agent_location, "target": self._target_location}
#
#     def _get_info(self):
#         return {
#             "distance": np.linalg.norm(
#                 self._agent_location - self._target_location, ord=1
#             )
#         }
#
#     def reset(self, seed=None, options=None):
#         # We need the following line to seed self.np_random
#         super().reset(seed=seed)
#
#         # Choose the agent's location uniformly at random
#         self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
#
#         # We will sample the target's location randomly until it does not
#         # coincide with the agent's location
#         self._target_location = self._agent_location
#         while np.array_equal(self._target_location, self._agent_location):
#             self._target_location = self.np_random.integers(
#                 0, self.size, size=2, dtype=int
#             )
#
#         observation = self._get_obs()
#         info = self._get_info()
#
#         if self.render_mode == "human":
#             self._render_frame()
#
#         return observation, info
#
#     def step(self, action):
#         # Map the action (element of {0,1,2,3}) to the direction we walk in
#         direction = self._action_to_direction[action]
#         # We use `np.clip` to make sure we don't leave the grid
#         self._agent_location = np.clip(
#             self._agent_location + direction, 0, self.size - 1
#         )
#         # An episode is done iff the agent has reached the target
#         terminated = np.array_equal(self._agent_location, self._target_location)
#         reward = 1 if terminated else 0  # Binary sparse rewards
#         observation = self._get_obs()
#         info = self._get_info()
#
#         if self.render_mode == "human":
#             self._render_frame()
#
#         return observation, reward, terminated, False, info
#
#     def render(self):
#         if self.render_mode == "rgb_array":
#             return self._render_frame()
#
#     def _render_frame(self):
#         if self.window is None and self.render_mode == "human":
#             pygame.init()
#             pygame.display.init()
#             self.window = pygame.display.set_mode((self.window_size, self.window_size))
#         if self.clock is None and self.render_mode == "human":
#             self.clock = pygame.time.Clock()
#
#         canvas = pygame.Surface((self.window_size, self.window_size))
#         canvas.fill((255, 255, 255))
#         pix_square_size = (
#             self.window_size / self.size
#         )  # The size of a single grid square in pixels
#
#         # First we draw the target
#         pygame.draw.rect(
#             canvas,
#             (255, 0, 0),
#             pygame.Rect(
#                 pix_square_size * self._target_location,
#                 (pix_square_size, pix_square_size),
#             ),
#         )
#         # Now we draw the agent
#         pygame.draw.circle(
#             canvas,
#             (0, 0, 255),
#             (self._agent_location + 0.5) * pix_square_size,
#             pix_square_size / 3,
#         )
#
#         # Finally, add some gridlines
#         for x in range(self.size + 1):
#             pygame.draw.line(
#                 canvas,
#                 0,
#                 (0, pix_square_size * x),
#                 (self.window_size, pix_square_size * x),
#                 width=3,
#             )
#             pygame.draw.line(
#                 canvas,
#                 0,
#                 (pix_square_size * x, 0),
#                 (pix_square_size * x, self.window_size),
#                 width=3,
#             )
#
#         if self.render_mode == "human":
#             # The following line copies our drawings from `canvas` to the visible window
#             self.window.blit(canvas, canvas.get_rect())
#             pygame.event.pump()
#             pygame.display.update()
#
#             # We need to ensure that human-rendering occurs at the predefined framerate.
#             # The following line will automatically add a delay to
#             # keep the framerate stable.
#             self.clock.tick(self.metadata["render_fps"])
#         else:  # rgb_array
#             return np.transpose(
#                 np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
#             )
#
#     def close(self):
#         if self.window is not None:
#             pygame.display.quit()
#             pygame.quit()
