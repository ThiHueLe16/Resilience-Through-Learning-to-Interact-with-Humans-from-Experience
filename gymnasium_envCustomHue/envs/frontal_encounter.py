import atexit
import math
import sys
import tempfile
from math import atan2
import os
import copy
import time
from typing import Dict, List, Tuple
import torch
import numpy as np
import yaml
from gymnasium import spaces
from SocNavGym.socnavgym.envs import SocNavEnv_v1
from SocNavGym.socnavgym.envs.socnavenv_v1 import SocNavGymObject
import random
from socnavgym.envs.socnavenv_v1 import SocNavEnv_v1, SocNavGymObject
import rvo2
from gymnasium import spaces
from shapely.geometry import Point, Polygon
from collections import namedtuple
import math
from math import ceil
from enum import Enum

from socnavgym.envs.utils.human import Human
from socnavgym.envs.utils.human_human import Human_Human_Interaction
from socnavgym.envs.utils.human_laptop import Human_Laptop_Interaction
from socnavgym.envs.utils.laptop import Laptop
from socnavgym.envs.utils.object import Object
from socnavgym.envs.utils.plant import Plant
from socnavgym.envs.utils.robot import Robot
from socnavgym.envs.utils.table import Table
from socnavgym.envs.utils.chair import Chair
from socnavgym.envs.utils.utils import (get_coordinates_of_rotated_rectangle,
                                        get_nearest_point_from_rectangle,
                                        get_square_around_circle,
                                        convert_angle_to_minus_pi_to_pi,
                                        compute_time_to_collision,
                                        point_to_segment_dist, w2px, w2py)
from socnavgym.envs.utils.wall import Wall
from socnavgym.envs.utils.sngnnv2.socnav_V2_API import SNScenario, SocNavAPI
from collections import namedtuple
EntityObs = namedtuple("EntityObs", ["id", "x", "y", "theta", "sin_theta", "cos_theta"])
DEBUG = 0
if 'debug' in sys.argv or "debug=2" in sys.argv:
    DEBUG = 2
elif "debug=1" in sys.argv:
    DEBUG = 1
MAX_ORIENTATION_CHANGE = math.pi / 2.

class FrontalEncounter(SocNavEnv_v1):
    # def __init__(self, socnavgym_env_config: str = None, custom_scenario_config: str = None, render_mode: str = None,
    #              scenario_index: int | None = None) -> None:
    def __init__(self, socnavgym_env_config: str=None, custom_scenario_config:str=None,render_mode: str = None)-> None:
        #Load custom scenario config
        with open(custom_scenario_config, "r") as custom_ymlfile:
            custom_config = yaml.safe_load(custom_ymlfile)

        with open(socnavgym_env_config, "r") as socnavenv_ymlfile:
            socnavenv_config = yaml.safe_load(socnavenv_ymlfile)
        self.mode= custom_config["scenario"]["mode"]
        # set custom aisle width in the socnavenvgym config. set robot, human start_y lane based on the alignment mode.
        if self.mode == "train":
            custom_min_width, custom_max_width= custom_config["scenario"]["aisle_width"]
            socnavenv_config["env"]["min_map_y"]= custom_min_width
            socnavenv_config["env"]["max_map_y"] = custom_max_width
            # set alignment mode for the custom env to use in the reset()
            self.alignment_mode = custom_config["scenario"]["alignment_mode"]
            # current_alignment mode is used to save the current mode chosen in training in set_y_robot
            self.current_alignment_mode=self.alignment_mode
        else: #in eval mode
            self.eval_scenario=custom_config["scenario"]["eval_scenarios"]
            self.num_eval_scenario=len(self.eval_scenario)
            # This index will be used by reset() when eval callback to load the next scenario
            self.current_eval_index=0

        # use self.y_r, self.y_h to log the y-coordinator of robot, and then later use y_r to set y_h during training. because y_r , y_h are sampled on different calls.
        # additinally, use y_r, y_h to set the goal of human and robot and human, which is sample on different calls.
        self.y_r, self.y_h = None, None
        self.d_path=None


        # write the modified socnaenv_config dict back to an .yaml file. otherwise error "TypeError: expected str, bytes or os.PathLike object, not dict" appear
        fd, tmp_path = tempfile.mkstemp(
            suffix=".yaml",
            prefix="frontal_encounter_cfg_"
        )
        with os.fdopen(fd, "w") as tmp_file:
            yaml.safe_dump(socnavenv_config, tmp_file)
        # Keep track so we can clean it up later
        self.tmp_config_path = tmp_path
        super().__init__(config=self.tmp_config_path, render_mode=render_mode)
        # use padding ->all wrapper will see fix-length vector. Otherwise SB3 will complain about shapes changing between episode
        # ( in our case: dynamic width-> wall shape vector change)
        # NO PADD CAUSE NOW THE WIDTH IS FIX
        # self.set_padded_observations(True)


        # self.is_entity_present["plants"] = False
        # self.is_entity_present["tables"] = False
        # self.is_entity_present["chairs"] = False
        # self.is_entity_present["laptops"] = False
        # self.is_entity_present["human_human_interactions"] = False
        # self.is_entity_present["human_laptop_interactions"] = False

        # self.alertness_value = None
        # self.safety_envelope_intervenes = False
        # register cleanup
        atexit.register(self.cleanup_temp_config)

    def cleanup_temp_config(self):
        try:
            if hasattr(self, "tmp_config_path") and os.path.exists(self.tmp_config_path):
                os.remove(self.tmp_config_path)
        except OSError:
            pass

    def set_y_robot(self, half_size_y, r_r,r_h, delta):
        """

        :param half_size_y:
        :param r_r: radius of robot
        :param r_h: radius of human
        :param delta: distance between 2 centers of human and robot
        :return: start y-coordinator of robot
        """
        y_r=None
        # set self.d_path for training
        if self.mode== "train":
            self.current_alignment_mode = random.choice(["aligned", "unaligned-top-bottom", "unaligned-bottom-top"])
            match self.current_alignment_mode:
                case "aligned":
                    self.d_path = 0.0
                case "unaligned-top-bottom":
                    while self.d_path == None or self.d_path == 0:
                        self.d_path = random.uniform(0, delta)
                case "unaligned-bottom-top":
                    while self.d_path == None or self.d_path == 0:
                        self.d_path = random.uniform(0, delta)

        # set y_r
        match self.current_alignment_mode:
            case "aligned":
                y_r=self.d_path
            case "unaligned-top-bottom":
                y_r= self.d_path/2
            case "unaligned-bottom-top":
                y_r = -self.d_path/2
        self.y_r= y_r
        return y_r

    def set_y_human(self, half_size_y, r_h, y_r, current_alignment_mode, delta):
        """
            #
            #     :param half_size_y:
            #     :param r_h: radius of human = HUMAN_DIAMETER/2
            #     :param y_r: y-coordinator of robot
            #     :return: initial y-coordinator of human based on alignment mode and robot's y-coordinator
            #     """
        y_h=None
        match self.current_alignment_mode:
            case "aligned":
                y_h = 0.0
            case "unaligned-top-bottom":
                y_h = -self.d_path/2
            case "unaligned-bottom-top":
                y_h = self.d_path/2
        self.y_h=y_h
        return y_h


    @staticmethod
    def sample_uniform(low: float, high: float) -> float:
        """
        :param low: lower range
        :param high: upper range
        :return: a random number in range [low , high] or [low, high) depend on rounding
        """
        """Simple helper with sanity check."""
        assert high > low, f"Invalid range: [{low}, {high}]"
        return random.uniform(low, high)



    def set_custom_goal_position(self, x_low, x_high, object_type: SocNavGymObject):
        x = random.uniform(x_low, x_high)
        y=None
        if object_type == SocNavGymObject.ROBOT:
            y=self.y_r
        elif object_type == SocNavGymObject.DYNAMIC_HUMAN:
            y=self.y_h
        return x, y

    def set_eval_custom_goal_position(self,object_type: SocNavGymObject):
        if object_type == SocNavGymObject.ROBOT:
            x = float(self.eval_scenario[self.current_eval_index]["x_goal_r"])
            y = self.y_r
        elif object_type == SocNavGymObject.DYNAMIC_HUMAN:
            x = float(self.eval_scenario[self.current_eval_index]["x_goal_h"])
            y = self.y_h
        return x, y


    def _get_kwargs(self, object_type: SocNavGymObject, extra_info: dict = None):
        """
            set start position of robot and human. custom this func to take control where to spawn robot and human
        """
        HALF_SIZE_X = self.MAP_X / 2. - self.MARGIN
        HALF_SIZE_Y = self.MAP_Y / 2. - self.MARGIN
        DELTA= self.ROBOT_RADIUS + self.HUMAN_DIAMETER / 2

        arg_dict = {}
        if object_type == SocNavGymObject.ROBOT:

            # custom x, y_position -FOR THESIS
            # if(self.mode=="eval"):
            #     #EVAL
            #     self.y_r=self.eval_scenario[self.current_eval_index]["robot_y"]
            # else:
            #     self.y_r= self.set_y_robot(HALF_SIZE_Y, self.ROBOT_RADIUS,self.HUMAN_DIAMETER/2,  DELTA)
            self.y_r = self.set_y_robot(HALF_SIZE_Y, self.ROBOT_RADIUS, self.HUMAN_DIAMETER / 2, DELTA)
            robot_start_y =self.y_r
            if self.mode=="eval":
                robot_start_x=float(self.eval_scenario[self.current_eval_index]["x_r"])
            else:
                robot_start_x = random.uniform(-HALF_SIZE_X+self.ROBOT_RADIUS+1, -HALF_SIZE_X+self.ROBOT_RADIUS + 1.3)

            arg_dict = {
                "id": 0,  # robot is assigned id 0
                "x":robot_start_x ,
                "y": robot_start_y,
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
            # custom x, y_position -FOR THESIS
            #  TRAIN
            self.y_h = self.set_y_human(HALF_SIZE_Y, self.HUMAN_DIAMETER/2, self.y_r, self.current_alignment_mode,DELTA)
            human_start_y=self.y_h
            if self.mode=="eval":
                human_start_x =float(self.eval_scenario[self.current_eval_index]["x_h"])
            else:
                human_start_x = random.uniform(HALF_SIZE_X-self.HUMAN_DIAMETER/2-1.3, HALF_SIZE_X-self.HUMAN_DIAMETER/2-1)
            arg_dict = {
                "id": self.id,
                "x": human_start_x,
                "y":human_start_y,
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
        elif object_type == SocNavGymObject.PLANT:
            arg_dict = {
                "id": self.id,
                "x": random.uniform(-HALF_SIZE_X, HALF_SIZE_X),
                "y": random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                "radius": self.PLANT_RADIUS + + random.uniform(-self.PLANT_RADIUS_MARGIN, self.PLANT_RADIUS_MARGIN)
            }
        elif object_type == SocNavGymObject.TABLE:
            arg_dict = {
                "id": self.id,
                "x": random.uniform(-HALF_SIZE_X, HALF_SIZE_X),
                "y": random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                "theta": random.uniform(-np.pi, np.pi),
                "width": self.TABLE_WIDTH + random.uniform(-self.TABLE_WIDTH_MARGIN, self.TABLE_WIDTH_MARGIN),
                "length": self.TABLE_LENGTH + random.uniform(-self.TABLE_LENGTH_MARGIN, self.TABLE_LENGTH_MARGIN)
            }
        elif object_type == SocNavGymObject.CHAIR:
            arg_dict = {
                "id": self.id,
                "x": random.uniform(-HALF_SIZE_X, HALF_SIZE_X),
                "y": random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                "theta": random.uniform(-np.pi, np.pi),
                "width": self.CHAIR_WIDTH + random.uniform(-self.CHAIR_WIDTH_MARGIN, self.CHAIR_WIDTH_MARGIN),
                "length": self.CHAIR_LENGTH + random.uniform(-self.CHAIR_LENGTH_MARGIN, self.CHAIR_LENGTH_MARGIN)
            }
        elif object_type == SocNavGymObject.LAPTOP:
            # pick a random table
            i = random.randint(0, len(self.tables) - 1)
            table = self.tables[i]

            # pick a random edge
            edge = np.random.randint(0, 4)
            if edge == 0:
                center = (
                    table.x + np.cos(table.orientation + np.pi / 2) * (table.width - self.LAPTOP_WIDTH) / 2,
                    table.y + np.sin(table.orientation + np.pi / 2) * (table.width - self.LAPTOP_WIDTH) / 2
                )
                theta = table.orientation + np.pi

            elif edge == 1:
                center = (
                    table.x + np.cos(table.orientation + np.pi) * (table.length - self.LAPTOP_LENGTH) / 2,
                    table.y + np.sin(table.orientation + np.pi) * (table.length - self.LAPTOP_LENGTH) / 2
                )
                theta = table.orientation - np.pi / 2

            elif edge == 2:
                center = (
                    table.x + np.cos(table.orientation - np.pi / 2) * (table.width - self.LAPTOP_WIDTH) / 2,
                    table.y + np.sin(table.orientation - np.pi / 2) * (table.width - self.LAPTOP_WIDTH) / 2
                )
                theta = table.orientation

            elif edge == 3:
                center = (
                    table.x + np.cos(table.orientation) * (table.length - self.LAPTOP_LENGTH) / 2,
                    table.y + np.sin(table.orientation) * (table.length - self.LAPTOP_LENGTH) / 2
                )
                theta = table.orientation + np.pi / 2

            arg_dict = {
                "id": self.id,
                "x": center[0],
                "y": center[1],
                "theta": theta,
                "width": self.LAPTOP_WIDTH,
                "length": self.LAPTOP_LENGTH
            }
        elif object_type == SocNavGymObject.HUMAN_HUMAN_INTERACTION_DYNAMIC \
                or object_type == SocNavGymObject.HUMAN_HUMAN_INTERACTION_DYNAMIC_NON_DISPERSING \
                or object_type == SocNavGymObject.HUMAN_HUMAN_INTERACTION_STATIC \
                or object_type == SocNavGymObject.HUMAN_HUMAN_INTERACTION_STATIC_NON_DISPERSING:
            assert extra_info != None
            if object_type == SocNavGymObject.HUMAN_HUMAN_INTERACTION_DYNAMIC:
                human_list = self.humans_in_h_h_dynamic_interactions
                interaction_type = "moving"
                can_disperse = True
            elif object_type == SocNavGymObject.HUMAN_HUMAN_INTERACTION_DYNAMIC_NON_DISPERSING:
                human_list = self.humans_in_h_h_dynamic_interactions_non_dispersing
                interaction_type = "moving"
                can_disperse = False
            elif object_type == SocNavGymObject.HUMAN_HUMAN_INTERACTION_STATIC:
                human_list = self.humans_in_h_h_static_interactions
                interaction_type = "stationary"
                can_disperse = True
            else:
                human_list = self.humans_in_h_h_static_interactions_non_dispersing
                interaction_type = "stationary"
                can_disperse = False
            index = extra_info["index"]
            arg_dict = {
                "x": random.uniform(-HALF_SIZE_X, HALF_SIZE_X),
                "y": random.uniform(-HALF_SIZE_Y, HALF_SIZE_Y),
                "type": interaction_type,
                "numOfHumans": human_list[index],
                "radius": self.INTERACTION_RADIUS,
                "human_width": self.HUMAN_DIAMETER,
                "MAX_HUMAN_SPEED": self.MAX_ADVANCE_HUMAN,
                "goal_radius": self.INTERACTION_GOAL_RADIUS,
                "noise": self.INTERACTION_NOISE_VARIANCE,
                "can_disperse": can_disperse,
                "pos_noise_std": self.HUMAN_POS_NOISE_STD,
                "angle_noise_std": self.HUMAN_ANGLE_NOISE_STD
            }
        elif object_type == SocNavGymObject.HUMAN_LAPTOP_INTERACTION \
                or object_type == SocNavGymObject.HUMAN_LAPTOP_INTERACTION_NON_DISPERSING:
            assert extra_info != None
            arg_dict = {
                "laptop": extra_info["laptop"],
                "distance": self.LAPTOP_WIDTH + self.HUMAN_LAPTOP_DISTANCE,
                "width": self.HUMAN_DIAMETER,
                "can_disperse": object_type == SocNavGymObject.HUMAN_LAPTOP_INTERACTION,
                "pos_noise_std": self.HUMAN_POS_NOISE_STD,
                "angle_noise_std": self.HUMAN_ANGLE_NOISE_STD

            }

        return arg_dict

    def sample_goal(self, goal_radius, object_type: SocNavGymObject, HALF_SIZE_X, HALF_SIZE_Y):
        #  OVERIDE FUNCTION FOR THESIS
        # adjust this code to change the end position of robot and human (suited the env for thesis project. other interaction(human-human goal is random like in the original code)
        # x will be on the half of the other side, y is on the line that go through start position of human and robot.
        start_time = time.time()

        while True:
            if self.check_timeout(start_time):
                break
            if object_type == SocNavGymObject.ROBOT:
                # robot_goal_x, robot_goal_y = FrontalEncounter.set_custom_goal_position(self.robot_start_x,self.robot_start_y,self.human_start_x, self.human_start_y,HALF_SIZE_X/2, HALF_SIZE_X)
                # orignal , before change
                # robot_goal_x, robot_goal_y =self.set_custom_goal_position(x_low=HALF_SIZE_X/2, x_high=HALF_SIZE_X-1,object_type = SocNavGymObject.ROBOT)
                if self.mode=="eval":
                    robot_goal_x, robot_goal_y=self.set_eval_custom_goal_position(object_type = SocNavGymObject.ROBOT)
                else:
                    robot_goal_x, robot_goal_y=self.set_custom_goal_position(x_low=HALF_SIZE_X/2, x_high=HALF_SIZE_X-2,object_type = SocNavGymObject.ROBOT)

                goal = Plant(
                    id=None,
                    x=robot_goal_x,
                    y=robot_goal_y,
                    radius=goal_radius
                )
            elif object_type == SocNavGymObject.DYNAMIC_HUMAN:
                # human_goal_x, human_goal_y = FrontalEncounter.set_custom_goal_position(self.robot_start_x,self.robot_start_y,self.human_start_x,self.human_start_y, -HALF_SIZE_X, -HALF_SIZE_X/2)
                if self.mode=="eval":
                    human_goal_x, human_goal_y=self.set_eval_custom_goal_position(object_type=SocNavGymObject.DYNAMIC_HUMAN)
                else:
                    human_goal_x, human_goal_y= self.set_custom_goal_position(x_low=-HALF_SIZE_X +1, x_high=-HALF_SIZE_X/2, object_type=SocNavGymObject.DYNAMIC_HUMAN)
                goal = Plant(
                    id=None,
                    x=human_goal_x,
                    y=human_goal_y,
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
            for obj in (all_objects + list(self.goals.values())):  # check if spawned object collides with any of the exisiting objects. It will not be rendered as a plant.
                if obj is None: continue
                # for robot/human goals, ignore collisions with robot & humans.because set up the goal to be on the line gotrough start huamn and robot start position, may colide. THESIS.
                if object_type in (SocNavGymObject.ROBOT, SocNavGymObject.DYNAMIC_HUMAN):
                    # if isinstance(obj, Robot) or isinstance(obj, Human):
                    continue
                #
                if (goal.collides(obj)):
                    collides = True
                    break
            if collides:
                del goal
            else:
                return goal
        return None

    def randomize_params(self):
        """
        To randomly initialize the number of entities of each type. Specifically, this function would initialize the MAP_SIZE, NUMBER_OF_HUMANS, NUMBER_OF_PLANTS, NUMBER_OF_LAPTOPS and NUMBER_OF_TABLES
        """
        self.MAP_X = random.uniform(self.MIN_MAP_X, self.MAX_MAP_X)
        # =============================================
        # ADJUST FOR THESIS.
        # EVAL: Adjust map_y to set width for eval environment
        # if self.mode=="eval":
        #     self.MAP_Y = self.eval_scenario[self.current_eval_index]["aisle_width"]
        # =============================================

        if self.shape == "square" or self.shape == "L":
            self.MAP_Y = self.MAP_X
        else:
            self.MAP_Y = random.uniform(self.MIN_MAP_Y, self.MAX_MAP_Y)

        self.ROBOT_RADIUS = self.INITIAL_ROBOT_RADIUS + random.uniform(-self.ROBOT_RADIUS_MARGIN,
                                                                       self.ROBOT_RADIUS_MARGIN)
        self.GOAL_RADIUS = self.INITIAL_GOAL_RADIUS + random.uniform(-self.GOAL_RADIUS_MARGIN, self.GOAL_RADIUS_MARGIN)
        self.GOAL_THRESHOLD = self.GOAL_RADIUS  # + self.ROBOT_RADIUS
        self.GOAL_ORIENTATION_THRESHOLD = random.uniform(self.MIN_GOAL_ORIENTATION_THRESHOLD,
                                                         self.MAX_GOAL_ORIENTATION_THRESHOLD)

        self.RESOLUTION_X = int(1850 * self.MAP_X / (self.MAP_X + self.MAP_Y))
        self.RESOLUTION_Y = int(1850 * self.MAP_Y / (self.MAP_X + self.MAP_Y))
        self.NUMBER_OF_STATIC_HUMANS = random.randint(self.MIN_STATIC_HUMANS,
                                                      self.MAX_STATIC_HUMANS)  # number of static humans in the env
        self.NUMBER_OF_DYNAMIC_HUMANS = random.randint(self.MIN_DYNAMIC_HUMANS,
                                                       self.MAX_DYNAMIC_HUMANS)  # number of static humans in the env
        self.NUMBER_OF_PLANTS = random.randint(self.MIN_PLANTS, self.MAX_PLANTS)  # number of plants in the env
        self.NUMBER_OF_TABLES = random.randint(self.MIN_TABLES, self.MAX_TABLES)  # number of tables in the env
        self.NUMBER_OF_CHAIRS = random.randint(self.MIN_CHAIRS, self.MAX_CHAIRS)  # number of chairs in the env
        self.NUMBER_OF_LAPTOPS = random.randint(self.MIN_LAPTOPS,
                                                self.MAX_LAPTOPS)  # number of laptops in the env. Laptops will be sampled on tables
        self.NUMBER_OF_H_H_DYNAMIC_INTERACTIONS = random.randint(self.MIN_H_H_DYNAMIC_INTERACTIONS,
                                                                 self.MAX_H_H_DYNAMIC_INTERACTIONS)  # number of dynamic human-human interactions
        self.NUMBER_OF_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING = random.randint(
            self.MIN_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING,
            self.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING)  # number of dynamic human-human interactions that do not disperse
        self.NUMBER_OF_H_H_STATIC_INTERACTIONS = random.randint(self.MIN_H_H_STATIC_INTERACTIONS,
                                                                self.MAX_H_H_STATIC_INTERACTIONS)  # number of static human-human interactions
        self.NUMBER_OF_H_H_STATIC_INTERACTIONS_NON_DISPERSING = random.randint(
            self.MIN_H_H_STATIC_INTERACTIONS_NON_DISPERSING,
            self.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING)  # number of static human-human interactions that do not disperse
        self.humans_in_h_h_dynamic_interactions = []
        self.humans_in_h_h_static_interactions = []
        self.humans_in_h_h_dynamic_interactions_non_dispersing = []
        self.humans_in_h_h_static_interactions_non_dispersing = []
        for _ in range(self.NUMBER_OF_H_H_DYNAMIC_INTERACTIONS):
            self.humans_in_h_h_dynamic_interactions.append(
                random.randint(self.MIN_HUMAN_IN_H_H_INTERACTIONS, self.MAX_HUMAN_IN_H_H_INTERACTIONS))
        for _ in range(self.NUMBER_OF_H_H_STATIC_INTERACTIONS):
            self.humans_in_h_h_static_interactions.append(
                random.randint(self.MIN_HUMAN_IN_H_H_INTERACTIONS, self.MAX_HUMAN_IN_H_H_INTERACTIONS))
        for _ in range(self.NUMBER_OF_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING):
            self.humans_in_h_h_dynamic_interactions_non_dispersing.append(
                random.randint(self.MIN_HUMAN_IN_H_H_INTERACTIONS, self.MAX_HUMAN_IN_H_H_INTERACTIONS))
        for _ in range(self.NUMBER_OF_H_H_STATIC_INTERACTIONS_NON_DISPERSING):
            self.humans_in_h_h_static_interactions_non_dispersing.append(
                random.randint(self.MIN_HUMAN_IN_H_H_INTERACTIONS, self.MAX_HUMAN_IN_H_H_INTERACTIONS))

        self.NUMBER_OF_H_L_INTERACTIONS = random.randint(self.MIN_H_L_INTERACTIONS,
                                                         self.MAX_H_L_INTERACTIONS)  # number of human laptop interactions
        self.NUMBER_OF_H_L_INTERACTIONS_NON_DISPERSING = random.randint(self.MIN_H_L_INTERACTIONS_NON_DISPERSING,
                                                                        self.MAX_H_L_INTERACTIONS_NON_DISPERSING)  # number of human laptop interactions that do not disperse
        self.TOTAL_H_L_INTERACTIONS = self.NUMBER_OF_H_L_INTERACTIONS + self.NUMBER_OF_H_L_INTERACTIONS_NON_DISPERSING

        # total humans
        self.total_humans = self.NUMBER_OF_STATIC_HUMANS + self.NUMBER_OF_DYNAMIC_HUMANS
        for i in self.humans_in_h_h_dynamic_interactions: self.total_humans += i
        for i in self.humans_in_h_h_static_interactions: self.total_humans += i
        for i in self.humans_in_h_h_dynamic_interactions_non_dispersing: self.total_humans += i
        for i in self.humans_in_h_h_static_interactions_non_dispersing: self.total_humans += i
        self.total_humans += self.TOTAL_H_L_INTERACTIONS
        # randomly select the shape
        if self.set_shape == "random":
            self.shape = random.choice(["rectangle", "square", "L"])
        else:
            self.shape = self.set_shape

        # adding Gaussian Noise to ORCA parameters
        self.orca_neighborDist = 2 * self.HUMAN_DIAMETER + np.random.randn()
        self.orca_timeHorizon = 5 + np.random.randn()
        self.orca_timeHorizonObst = 5 + np.random.randn()
        self.orca_maxSpeed = self.MAX_ADVANCE_HUMAN + np.random.randn() * 0.01

        # adding Gaussian Noise to SFM parameters
        self.sfm_r0 = abs(0.05 + np.random.randn() * 0.01)
        self.sfm_gamma = 0.25 + np.random.randn() * 0.01
        self.sfm_n = 1 + np.random.randn() * 0.1
        self.sfm_n_prime = 1 + np.random.randn() * 0.1
        self.sfm_lambd = 1 + np.random.randn() * 0.1



    def try_reset(self, seed=None, options=None):
        """
        Resets the environment
        """
        start_time = time.time()
        if not self.has_configured:
            raise Exception("Please pass in the keyword argument config=\"path to config\" while calling gym.make")
        self.cumulative_reward = 0

        # setting seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # randomly initialize the parameters
        self.randomize_params()
        self.id = 1

        # EVAL, set MAP_Y, d_path, THESIS
        if self.mode == "eval":
            # self.MAP_Y=self.eval_scenario[self.current_eval_index]["aisle_width"]
            self.d_path = float(self.eval_scenario[self.current_eval_index]["d_path"])
            if self.d_path <0:
                raise ValueError(f"d_path must be > 0 in eval config, got {self.d_path}")
            self.current_alignment_mode = self.eval_scenario[self.current_eval_index]["alignment_mode"]
        else:
            self.d_path = None

        HALF_SIZE_X = self.MAP_X / 2. - self.MARGIN
        HALF_SIZE_Y = self.MAP_Y / 2. - self.MARGIN

        # reset the start_y human and robot
        # self.y_r = self.set_y_robot(HALF_SIZE_Y)
        # print(f" robot-human start y: {self.y_r, self.y_h}")
        # self.current_alignment_mode=None

        # keeping track of the scenarios for sngnn reward
        self.sn_sequence = []

        # to keep track of the current objects
        self.objects = []
        self.laptops = []
        self.walls = []
        self.static_humans = []
        self.dynamic_humans = []
        self.plants = []
        self.tables = []
        self.chairs = []
        self.goals: Dict[
            int, Plant] = {}  # dictionary to store all the goals. The key would be the id of the entity. The goal would be a Plant object so that collision checks can be done.
        self.moving_interactions = []  # a list to keep track of moving interactions
        self.static_interactions = []
        self.h_l_interactions = []

        # clearing img_list
        if self.img_list is not None:
            del self.img_list
            self.img_list = None

        # variable that shows whether a crowd is being formed currently or not
        self.crowd_forming = False

        # variable that shows whether a human-laptop-interaction is being formed or not
        self.h_l_forming = False

        # adding walls to the environment
        self._add_walls()

        # robot
        robot = self._sample_object(start_time, SocNavGymObject.ROBOT)
        if robot == None:
            return False, None, None
        self.robot = robot
        self.objects.append(self.robot)

        # making a copy of the robot for calculating time taken by a robot that has orca policy
        self.robot_orca = copy.deepcopy(self.robot)
        # defining a few parameters for the orca robot
        self.has_orca_robot_reached_goal = False
        self.has_orca_robot_collided = False
        self.orca_robot_reach_time = None
        self.orca_robot_path_length = 0

        # dynamic humans
        for _ in range(self.NUMBER_OF_DYNAMIC_HUMANS):  # spawn specified number of humans
            human = self._sample_object(start_time, SocNavGymObject.DYNAMIC_HUMAN)
            if human == None:
                return False, None, None
            self.dynamic_humans.append(human)
            self.objects.append(human)
            self.id += 1

        # static humans
        for _ in range(self.NUMBER_OF_STATIC_HUMANS):  # spawn specified number of humans
            human = self._sample_object(start_time, SocNavGymObject.STATIC_HUMAN)
            if human == None:
                return False, None, None
            self.static_humans.append(human)
            self.objects.append(human)
            self.id += 1

        # plants
        for _ in range(self.NUMBER_OF_PLANTS):  # spawn specified number of plants
            plant = self._sample_object(start_time, SocNavGymObject.PLANT)
            if plant == None:
                return False, None, None
            self.plants.append(plant)
            self.objects.append(plant)
            self.id += 1

        # tables
        for _ in range(self.NUMBER_OF_TABLES):  # spawn specified number of tables
            table = self._sample_object(start_time, SocNavGymObject.TABLE)
            if table == None:
                return False, None, None
            self.tables.append(table)
            self.objects.append(table)
            self.id += 1

        # chairs
        for _ in range(self.NUMBER_OF_CHAIRS):  # spawn specified number of chairs
            chair = self._sample_object(start_time, SocNavGymObject.CHAIR)
            if chair == None:
                return False, None, None
            self.chairs.append(chair)
            self.objects.append(chair)
            self.id += 1

        # laptops
        if (len(self.tables) == 0):
            pass
        elif self.NUMBER_OF_LAPTOPS + self.NUMBER_OF_H_L_INTERACTIONS + self.NUMBER_OF_H_L_INTERACTIONS_NON_DISPERSING > 4 * self.NUMBER_OF_TABLES:
            raise AssertionError("Number of laptops exceeds the number of edges available on tables")
        else:
            for _ in range(self.NUMBER_OF_LAPTOPS):  # placing laptops on tables
                laptop = self._sample_object(start_time, SocNavGymObject.LAPTOP)
                if laptop == None:
                    return False, None, None
                self.laptops.append(laptop)
                self.objects.append(laptop)
                self.id += 1

        # interactions
        for ind in range(self.NUMBER_OF_H_H_DYNAMIC_INTERACTIONS):
            i = self._sample_object(start_time, SocNavGymObject.HUMAN_HUMAN_INTERACTION_DYNAMIC,
                                    extra_info={"index": ind})
            if i == None:
                return False, None, None
            self.moving_interactions.append(i)
            self.objects.append(i)
            for human in i.humans:
                human.id = self.id
                self.id += 1

        for ind in range(self.NUMBER_OF_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING):
            i = self._sample_object(start_time, SocNavGymObject.HUMAN_HUMAN_INTERACTION_DYNAMIC_NON_DISPERSING,
                                    extra_info={"index": ind})
            if i == None:
                return False, None, None
            self.moving_interactions.append(i)
            self.objects.append(i)
            for human in i.humans:
                human.id = self.id
                self.id += 1

        for ind in range(self.NUMBER_OF_H_H_STATIC_INTERACTIONS):
            i = self._sample_object(start_time, SocNavGymObject.HUMAN_HUMAN_INTERACTION_STATIC,
                                    extra_info={"index": ind})
            if i == None:
                return False, None, None
            self.static_interactions.append(i)
            self.objects.append(i)
            for human in i.humans:
                human.id = self.id
                self.id += 1

        for ind in range(self.NUMBER_OF_H_H_STATIC_INTERACTIONS_NON_DISPERSING):
            i = self._sample_object(start_time, SocNavGymObject.HUMAN_HUMAN_INTERACTION_STATIC_NON_DISPERSING,
                                    extra_info={"index": ind})
            if i == None:
                return False, None, None
            self.static_interactions.append(i)
            self.objects.append(i)
            for human in i.humans:
                human.id = self.id
                self.id += 1

        for _ in range(self.NUMBER_OF_H_L_INTERACTIONS):
            # sampling a laptop
            laptop, interaction = self._sample_human_laptop_interaction(start_time,
                                                                        SocNavGymObject.HUMAN_LAPTOP_INTERACTION)
            if laptop == None or interaction == None:
                return False, None, None
            self.h_l_interactions.append(interaction)
            self.objects.append(interaction)
            self.id += 1
            interaction.human.id = self.id
            self.id += 1

        for _ in range(self.NUMBER_OF_H_L_INTERACTIONS_NON_DISPERSING):
            # sampling a laptop
            laptop, interaction = self._sample_human_laptop_interaction(start_time,
                                                                        SocNavGymObject.HUMAN_LAPTOP_INTERACTION_NON_DISPERSING)
            if laptop == None or interaction == None:
                return False, None, None
            self.h_l_interactions.append(interaction)
            self.objects.append(interaction)
            self.id += 1
            interaction.human.id = self.id
            self.id += 1

        # assigning ids to walls
        for wall in self.walls:
            wall.id = self.id
            self.id += 1

        # adding goals
        # important TO set the goal position
        for human in self.dynamic_humans:
            o = self.sample_goal(self.HUMAN_GOAL_RADIUS, SocNavGymObject.DYNAMIC_HUMAN, HALF_SIZE_X, HALF_SIZE_Y)
            if o is None:
                return False, None, None
            self.goals[human.id] = o
            human.set_goal(o.x, o.y)

        for human in self.static_humans:
            self.goals[human.id] = Plant(id=None, x=human.x, y=human.y, radius=self.HUMAN_GOAL_RADIUS)
            human.set_goal(human.x, human.y)  # setting goal of static humans to where they are spawned

        robot_goal = self.sample_goal(self.GOAL_RADIUS, SocNavGymObject.ROBOT, HALF_SIZE_X, HALF_SIZE_Y)
        if robot_goal is None:
            return False, None, None
        self.goals[self.robot.id] = robot_goal
        self.robot.goal_x = robot_goal.x
        self.robot.goal_y = robot_goal.y
        self.robot.goal_a = random.uniform(-np.pi, np.pi)
        self.robot_orca.goal_x = robot_goal.x
        self.robot_orca.goal_y = robot_goal.y
        self.robot_orca.goal_a = self.robot.goal_a
        for i in self.moving_interactions:
            o = self.sample_goal(self.INTERACTION_GOAL_RADIUS, HALF_SIZE_X, HALF_SIZE_Y)
            if o is None:
                return False, None, None
            for human in i.humans:
                self.goals[human.id] = o
            i.set_goal(o.x, o.y)

        self._is_terminated = False
        self._is_truncated = False
        self._collision = False
        self.ticks = 0
        self.compliant_count = 0  # keeps track of how many times the agent is outside the personal space of humans
        self.prev_goal_distance = np.sqrt(
            (self.robot.x - self.robot.goal_x) ** 2 + (self.robot.y - self.robot.goal_y) ** 2)
        self.robot_path_length = 0
        self.stalled_time = 0
        self.failure_to_progress = 0
        self.v_min = float("inf")
        self.v_max = 0.0
        self.v_avg = 0.0
        self.prev_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.a_min = float("inf")
        self.a_max = 0.0
        self.a_avg = 0.0
        self.prev_a = np.array([0.0, 0.0], dtype=np.float32)
        self.jerk_min = float("inf")
        self.jerk_max = 0.0
        self.jerk_avg = 0.0

        # all entities in the environment
        self.count = 0

        # a dictionary indexed by the id of the entity that stores the previous state observations for all the entities (except walls)
        self._prev_observations: Dict[int, EntityObs] = {}
        self._current_observations: Dict[int, EntityObs] = {}
        self.populate_prev_obs()

        obs = self._get_obs()

        self.reward_calculator.re_init(self)
        if self.reward_calculator.use_sngnn:
            self.reward_calculator.sngnn = SocNavAPI(
                device=('cuda' + str(self.cuda_device) if torch.cuda.is_available() else 'cpu'), params_dir=(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "sngnnv2", "example_model")))

        # EVAL. update index, next reset()-> next scenario
        if self.mode== "eval":
            self.current_eval_index= (self.current_eval_index+1)%self.num_eval_scenario
        return True, obs, {}

    def step(self, action_pre):
        """Computes a step in the current episode given the action.

        Args:
            action_pre (Union[numpy.ndarray, list]): An action that lies in the action space

        Returns:
            observation (numpy.ndarray) : the observation from the current action
            reward (float) : reward received on the current action
            terminated (bool) : whether the episode has finished or not
            truncated (bool) : whether the episode has finished due to time limit or not
            info (dict) : additional information
        """

        # for converting the action to the velocity
        def process_action(act):
            """Converts the values from [-1,1] to the corresponding velocity values

            Args:
                act (np.ndarray): action from the action space

            Returns:
                np.ndarray: action with velocity values
            """
            action = act.astype(np.float32)
            # action[0] = (float(action[0]+1.0)/2.0)*self.MAX_ADVANCE_ROBOT   # [-1, +1] --> [0, self.MAX_ADVANCE_ROBOT]
            action[0] = ((action[0] + 0.0) / 1.0) * self.MAX_ADVANCE_ROBOT  # [-1, +1] --> [-MAX_ADVANCE, +MAX_ADVANCE]
            if action[1] != 0.0 and self.robot.type == "diff-drive": raise AssertionError(
                "Differential Drive robot cannot have lateral speed")
            action[1] = ((action[1] + 0.0) / 1.0) * self.MAX_ADVANCE_ROBOT  # [-1, +1] --> [-MAX_ADVANCE, +MAX_ADVANCE]
            action[2] = (float(
                action[2] + 0.0) / 1.0) * self.MAX_ROTATION  # [-1, +1] --> [-self.MAX_ROTATION, +self.MAX_ROTATION]
            # if action[0] < 0:               # Advance must be positive
            #     action[0] *= -1
            if action[0] > self.MAX_ADVANCE_ROBOT:  # Advance must be less or equal self.MAX_ADVANCE_ROBOT
                action[0] = self.MAX_ADVANCE_ROBOT
            if action[0] < -self.MAX_ADVANCE_ROBOT:  # Advance must be less or equal self.MAX_ADVANCE_ROBOT
                action[0] = -self.MAX_ADVANCE_ROBOT
            if action[1] > self.MAX_ADVANCE_ROBOT:  # Advance must be less or equal self.MAX_ADVANCE_ROBOT
                action[1] = self.MAX_ADVANCE_ROBOT
            if action[1] < -self.MAX_ADVANCE_ROBOT:  # Advance must be less or equal self.MAX_ADVANCE_ROBOT
                action[1] = -self.MAX_ADVANCE_ROBOT
            if action[2] < -self.MAX_ROTATION:  # Rotation must be higher than -self.MAX_ROTATION
                action[2] = -self.MAX_ROTATION
            elif action[2] > +self.MAX_ROTATION:  # Rotation must be lower than +self.MAX_ROTATION
                action[2] = +self.MAX_ROTATION
            return action

        # if action is a list, converting it to numpy.ndarray
        if (type(action_pre) == list):
            action_pre = np.array(action_pre, dtype=np.float32)

        # call error if the environment wasn't reset after the episode ended
        if self._is_truncated or self._is_terminated:
            raise Exception('step call within a finished episode!')

        # calculating the velocity from action
        action = process_action(action_pre)

        # setting the robot's velocities
        self.robot.vel_x = action[0]
        self.robot.vel_y = action[1]
        self.robot.vel_a = action[2]

        # update robot
        if self._collision:
            future_robot = copy.deepcopy(self.robot)
            future_robot.update(self.TIMESTEP)
            collision_human, collision_object, collision_wall = self.check_robot_collision(future_robot)
            collision = collision_human or collision_object or collision_wall
            execute_action = not collision
        else:
            execute_action = True

        if execute_action:
            self.robot.update(self.TIMESTEP)

        # update dummy robot with orca policy
        if (not self.has_orca_robot_collided) and (not self.has_orca_robot_reached_goal):
            vel = self.compute_orca_velocity_robot(self.robot_orca)
            if self.robot_orca.type == "holonomic":
                vel_x = vel[0] * np.cos(self.robot_orca.orientation) + vel[1] * np.sin(self.robot_orca.orientation)
                vel_y = -vel[0] * np.sin(self.robot_orca.orientation) + vel[1] * np.cos(self.robot_orca.orientation)
                vel_a = (np.arctan2(vel[1], vel[0]) - self.robot_orca.orientation) / self.TIMESTEP
            elif self.robot_orca.type == "diff-drive":
                vel_y = 0
                vel_a = (np.arctan2(vel[1], vel[0]) - self.robot_orca.orientation) / self.TIMESTEP
                vel_x = np.sqrt(vel[0] ** 2 + vel[1] ** 2)

            self.robot_orca.vel_x = np.clip(vel_x, -self.MAX_ADVANCE_ROBOT, self.MAX_ADVANCE_ROBOT)
            self.robot_orca.vel_y = np.clip(vel_y, -self.MAX_ADVANCE_ROBOT, self.MAX_ADVANCE_ROBOT)
            self.robot_orca.vel_a = np.clip(vel_a, -self.MAX_ROTATION, self.MAX_ROTATION)
            self.robot_orca.update(self.TIMESTEP)

        # update humans
        interaction_vels = self.compute_orca_interaction_velocities()
        for index, human in enumerate(self.dynamic_humans):
            if (human.goal_x == None or human.goal_y == None):
                raise AssertionError("Human goal not specified")
            if human.policy == "orca":
                velocity = self.compute_orca_velocity(human)
            elif human.policy == "sfm":
                velocity = self.compute_sfm_velocity(human)

            orientation = atan2(velocity[1], velocity[0])
            if human.set_new_orientation_with_limits(orientation, MAX_ORIENTATION_CHANGE, self.TIMESTEP):
                human.speed = np.linalg.norm(velocity)
                if human.speed < self.SPEED_THRESHOLD and not (
                        self.crowd_forming and human.id in self.humans_forming_crowd.keys()): human.speed = 0
            else:
                human.speed = 0
            # human.update(self.TIMESTEP)

        # updating moving humans in interactions
        for index, i in enumerate(self.moving_interactions):
            i.update_speed(self.TIMESTEP, interaction_vels[index], self.MAX_ROTATION_HUMAN)

        # update the goals for humans if they have reached goal
        for human in self.dynamic_humans:
            if self.crowd_forming and human.id in self.humans_forming_crowd.keys(): continue  # handling the humans forming a crowd separately
            if self.h_l_forming and human.id == self.h_l_forming_human.id: continue  # handling the human forming the human-laptop-interaction separately
            HALF_SIZE_X = self.MAP_X / 2. - self.MARGIN
            HALF_SIZE_Y = self.MAP_Y / 2. - self.MARGIN
            if human.has_reached_goal():
                # adjust this below for THESIS. because i change the sample_goal method
                o = self.sample_goal(self.HUMAN_GOAL_RADIUS,SocNavGymObject.DYNAMIC_HUMAN, HALF_SIZE_X, HALF_SIZE_Y)
                if o is not None:
                    human.set_goal(o.x, o.y)
                    self.goals[human.id] = o

        # update goals of interactions
        for i in self.moving_interactions:
            if i.has_reached_goal():
                HALF_SIZE_X = self.MAP_X / 2. - self.MARGIN
                HALF_SIZE_Y = self.MAP_Y / 2. - self.MARGIN
                o = self.sample_goal(self.INTERACTION_GOAL_RADIUS, HALF_SIZE_X, HALF_SIZE_Y)
                if o is not None:
                    i.set_goal(o.x, o.y)
                    for human in i.humans:
                        self.goals[human.id] = o

        # complete the crowd formation if all the crowd-forming humans have reached their goals
        if self.crowd_forming:  # enter only when the environment is undergoing a crowd formation
            haveAllHumansReached = True
            for human in self.humans_forming_crowd.values():
                if human.has_reached_goal(offset=0):
                    # updating the orientation of humans so that the humans look towards each other
                    human.orientation = self.upcoming_interaction.humans[self.id_to_index[human.id]].orientation
                else:
                    haveAllHumansReached = False
            if haveAllHumansReached:
                self.finish_human_crowd_formation()
            else:
                if self.check_almost_crowd_formed():
                    self.almost_formed_crowd_count += 1
                else:
                    self.almost_formed_crowd_count = 0

                if self.almost_formed_crowd_count == 25:
                    self.finish_human_crowd_formation(make_approx_crowd=True)

        # complete human laptop interaction formation if the human has reached goal
        if self.h_l_forming:
            if self.h_l_forming_human.has_reached_goal(offset=0):
                self.finish_h_l_formation()

        # handling collisions
        self.handle_collision_and_update()

        # getting observations
        observation = self._get_obs()

        # computing rewards and done
        reward, info = self.compute_reward_and_ticks(action)
        terminated = self._is_terminated
        truncated = self._is_truncated

        # updating the previous observations
        self.populate_prev_obs()

        self.cumulative_reward += reward

        # providing debugging information
        if DEBUG > 0 and self.ticks % 50 == 0:
            self.render()
        elif DEBUG > 1:
            self.render()

        if DEBUG > 0 and (self._is_terminated or self._is_truncated):
            print(f'cumulative reward: {self.cumulative_reward}')

        # dispersing crowds
        if np.random.random() <= self.CROWD_DISPERSAL_PROBABILITY:
            t = np.random.randint(0, 2)
            self.dispersable_moving_crowd_indices = []
            self.dispersable_static_crowd_indices = []

            for ind, i in enumerate(self.moving_interactions):
                if i.can_disperse:
                    self.dispersable_moving_crowd_indices.append(ind)

            for ind, i in enumerate(self.static_interactions):
                if i.can_disperse:
                    self.dispersable_static_crowd_indices.append(ind)

            if t == 0 and len(self.dispersable_static_crowd_indices) > 0:
                index = random.choice(self.dispersable_static_crowd_indices)
                self.disperse_static_crowd(index)

            elif t == 1 and len(self.dispersable_moving_crowd_indices) > 0:
                index = random.choice(self.dispersable_moving_crowd_indices)
                self.disperse_moving_crowd(index)

        # disperse human-laptop
        self.dispersable_h_l_interaction_indices = []
        for ind, i in enumerate(self.h_l_interactions):
            if i.can_disperse:
                self.dispersable_h_l_interaction_indices.append(ind)

        if np.random.random() <= self.HUMAN_LAPTOP_DISPERSAL_PROBABILITY and len(
                self.dispersable_h_l_interaction_indices) > 0:
            index = random.choice(self.dispersable_h_l_interaction_indices)
            self.disperse_human_laptop(index)

        # forming interactions
        if np.random.random() <= self.CROWD_FORMATION_PROBABILITY and not self.crowd_forming and not self.h_l_forming:
            self.form_human_crowd()  # form a new human crowd

        if np.random.random() <= self.HUMAN_LAPTOP_FORMATION_PROBABILITY and not self.crowd_forming and not self.h_l_forming:
            self.form_human_laptop_interaction()  # form a new human-laptop interaction

        return observation, reward, terminated, truncated, info

    @property
    def observation_space(self):
        """
        Observation space includes the goal coordinates in the robot's frame and the relative coordinates and speeds (linear & angular) of all the objects in the scenario

        Returns:
        gym.spaces.Dict : the observation space of the environment
        """
        # THESIS: override method from the original Socnavenv_v1, change the boundary of x, y relative of human in frame of robot. because the custom env force robot and human to appear initially opposite
        # in opposite side -> initial distance of human relative to robot is bigger than MAP_Y*SQRT(2), speacially when length is bigger than width
        # ======
        max_relative=np.sqrt(self.MAP_X**2 + self.MAP_Y**2)
        low_rel=-max_relative
        high_rel= max_relative
        # ===
        d = {

            "robot": spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0,low_rel , low_rel, -self.ROBOT_RADIUS],
                             dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, high_rel, high_rel, self.ROBOT_RADIUS],
                              dtype=np.float32),
                shape=((self.robot_obs_dim,)),
                dtype=np.float32

            )
        }
        max_relative = np.sqrt(self.MAP_X ** 2 + self.MAP_Y ** 2)
        low_rel = -max_relative
        high_rel = max_relative
        if self.is_entity_present["humans"]:
            # THESIS: redefine the boundary of the value field human's x-, y-relate to robot in robot frame
            d["humans"] = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, low_rel, low_rel, -1.0, -1.0,
                              -self.HUMAN_DIAMETER / 2, -(self.MAX_ADVANCE_HUMAN + self.MAX_ADVANCE_ROBOT) * np.sqrt(2),
                              -2 * np.pi / self.TIMESTEP, 0] * ((self.MAX_HUMANS + (
                            self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING) + ((self.MAX_H_H_DYNAMIC_INTERACTIONS + self.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING) * self.MAX_HUMAN_IN_H_H_INTERACTIONS) + (
                                                                             (
                                                                                         self.MAX_H_H_STATIC_INTERACTIONS + self.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING) * self.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.get_padded_observations else self.total_humans),
                             dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1, high_rel, high_rel, 1.0, 1.0,
                               self.HUMAN_DIAMETER / 2, +(self.MAX_ADVANCE_HUMAN + self.MAX_ADVANCE_ROBOT) * np.sqrt(2),
                               +2 * np.pi / self.TIMESTEP, 1] * ((self.MAX_HUMANS + (
                            self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING) + ((
                                                                                                                 self.MAX_H_H_DYNAMIC_INTERACTIONS + self.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING) * self.MAX_HUMAN_IN_H_H_INTERACTIONS) + (
                                                                              (
                                                                                          self.MAX_H_H_STATIC_INTERACTIONS + self.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING) * self.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.get_padded_observations else self.total_humans),
                              dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 8) * ((self.MAX_HUMANS + (
                            self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING) + ((
                                                                                                                 self.MAX_H_H_DYNAMIC_INTERACTIONS + self.MAX_H_H_DYNAMIC_INTERACTIONS_NON_DISPERSING) * self.MAX_HUMAN_IN_H_H_INTERACTIONS) + (
                                                                                   (
                                                                                               self.MAX_H_H_STATIC_INTERACTIONS + self.MAX_H_H_STATIC_INTERACTIONS_NON_DISPERSING) * self.MAX_HUMAN_IN_H_H_INTERACTIONS)) if self.get_padded_observations else self.total_humans),)),
                dtype=np.float32
            )

        if self.is_entity_present["laptops"]:
            d["laptops"] = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0,
                              -self.LAPTOP_RADIUS, -(self.MAX_ADVANCE_ROBOT) * np.sqrt(2), -self.MAX_ROTATION, 0] * ((
                                                                                                                                 self.MAX_LAPTOPS + (
                                                                                                                                     self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING)) if self.get_padded_observations else (
                            self.NUMBER_OF_LAPTOPS + self.TOTAL_H_L_INTERACTIONS)), dtype=np.float32),
                high=np.array(
                    [1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, self.LAPTOP_RADIUS,
                     +(self.MAX_ADVANCE_ROBOT) * np.sqrt(2), +self.MAX_ROTATION, 1] * ((self.MAX_LAPTOPS + (
                                self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING)) if self.get_padded_observations else (
                                self.NUMBER_OF_LAPTOPS + self.TOTAL_H_L_INTERACTIONS)), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 8) * ((self.MAX_LAPTOPS + (
                            self.MAX_H_L_INTERACTIONS + self.MAX_H_L_INTERACTIONS_NON_DISPERSING)) if self.get_padded_observations else (
                            self.NUMBER_OF_LAPTOPS + self.TOTAL_H_L_INTERACTIONS)),)),
                dtype=np.float32

            )

        if self.is_entity_present["tables"]:
            d["tables"] = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0,
                              -self.TABLE_RADIUS, -(self.MAX_ADVANCE_ROBOT) * np.sqrt(2), -self.MAX_ROTATION, 0] * (
                                 self.MAX_TABLES if self.get_padded_observations else self.NUMBER_OF_TABLES),
                             dtype=np.float32),
                high=np.array(
                    [1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, self.TABLE_RADIUS,
                     +(self.MAX_ADVANCE_ROBOT) * np.sqrt(2), +self.MAX_ROTATION, 1] * (
                        self.MAX_TABLES if self.get_padded_observations else self.NUMBER_OF_TABLES), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 8) * (
                    self.MAX_TABLES if self.get_padded_observations else self.NUMBER_OF_TABLES),)),
                dtype=np.float32

            )

        if self.is_entity_present["plants"]:
            d["plants"] = spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0,
                              -self.PLANT_RADIUS, -(self.MAX_ADVANCE_ROBOT) * np.sqrt(2), -self.MAX_ROTATION, 0] * (
                                 self.MAX_PLANTS if self.get_padded_observations else self.NUMBER_OF_PLANTS),
                             dtype=np.float32),
                high=np.array(
                    [1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0, self.PLANT_RADIUS,
                     +(self.MAX_ADVANCE_ROBOT) * np.sqrt(2), +self.MAX_ROTATION, 1] * (
                        self.MAX_PLANTS if self.get_padded_observations else self.NUMBER_OF_PLANTS), dtype=np.float32),
                shape=(((self.robot.one_hot_encoding.shape[0] + 8) * (
                    self.MAX_PLANTS if self.get_padded_observations else self.NUMBER_OF_PLANTS),)),
                dtype=np.float32

            )

        if not self.get_padded_observations:
            total_segments = 0
            for w in self.walls:
                total_segments += w.length // self.WALL_SEGMENT_SIZE
                if w.length % self.WALL_SEGMENT_SIZE != 0: total_segments += 1
            if self.is_entity_present["walls"]:
                d["walls"] = spaces.Box(
                    low=np.array([0, 0, 0, 0, 0, 0, -self.MAP_X * np.sqrt(2), -self.MAP_Y * np.sqrt(2), -1.0, -1.0,
                                  -self.WALL_SEGMENT_SIZE, -(self.MAX_ADVANCE_ROBOT) * np.sqrt(2), -self.MAX_ROTATION,
                                  0] * int(total_segments), dtype=np.float32),
                    high=np.array([1, 1, 1, 1, 1, 1, +self.MAP_X * np.sqrt(2), +self.MAP_Y * np.sqrt(2), 1.0, 1.0,
                                   +self.WALL_SEGMENT_SIZE, +(self.MAX_ADVANCE_ROBOT) * np.sqrt(2), +self.MAX_ROTATION,
                                   1] * int(total_segments), dtype=np.float32),
                    shape=(((self.robot.one_hot_encoding.shape[0] + 8) * int(total_segments),)),
                    dtype=np.float32
                )

        return spaces.Dict(d)


    def compute_reward_and_ticks(self, action):
        """
        Function to compute the reward and also calculate if the episode has finished
        """
        self.ticks += 1

        # calculate the distance to the goal
        distance_to_goal = np.sqrt((self.robot.goal_x - self.robot.x) ** 2 + (self.robot.goal_y - self.robot.y) ** 2)
        diff_goal_angle = abs(np.arctan2(np.sin(self.robot.goal_a), np.cos(self.robot.goal_a)) - np.arctan2(
            np.sin(self.robot.orientation), np.cos(self.robot.orientation)))

        # calculate the distance to goal for the orca robot
        if (not self.has_orca_robot_collided) and (not self.has_orca_robot_reached_goal):
            distance_to_goal_orca_robot = np.sqrt(
                (self.robot_orca.goal_x - self.robot_orca.x) ** 2 + (self.robot_orca.goal_y - self.robot_orca.y) ** 2)
            # check for object-robot collisions
            orca_robot_collision = False

            for object in self.static_humans + self.dynamic_humans + self.plants + self.walls + self.tables + self.chairs + self.laptops:
                if (self.robot_orca.collides(object)):
                    orca_robot_collision = True

            # interaction-robot collision
            for i in (self.moving_interactions + self.static_interactions + self.h_l_interactions):
                if orca_robot_collision:
                    break
                if i.collides(self.robot):
                    orca_robot_collision = True
                    break

            if orca_robot_collision:
                self.has_orca_robot_collided = True

            if distance_to_goal_orca_robot < self.GOAL_THRESHOLD:
                self.has_orca_robot_reached_goal = True
                self.orca_robot_reach_time = self.ticks

            orca_robot_speed = np.linalg.norm(
                np.array([self.robot_orca.vel_x, self.robot_orca.vel_y], dtype=np.float32))
            self.orca_robot_path_length += orca_robot_speed * self.TIMESTEP

        collision_human, collision_object, collision_wall = self.check_robot_collision(self.robot)

        collision = collision_object or collision_human

        dmin = float('inf')

        self.all_humans = []
        for human in self.static_humans + self.dynamic_humans: self.all_humans.append(human)

        for i in self.static_interactions + self.moving_interactions:
            for h in i.humans: self.all_humans.append(h)

        for i in self.h_l_interactions: self.all_humans.append(i.human)

        for human in self.all_humans:
            px = human.x - self.robot.x
            py = human.y - self.robot.y

            vx = human.speed * np.cos(human.orientation) - action[0] * np.cos(
                action[2] * self.TIMESTEP + self.robot.orientation) - action[1] * np.cos(
                action[2] * self.TIMESTEP + self.robot.orientation + np.pi / 2)
            vy = human.speed * np.sin(human.orientation) - action[0] * np.sin(
                action[2] * self.TIMESTEP + self.robot.orientation) - action[1] * np.sin(
                action[2] * self.TIMESTEP + self.robot.orientation + np.pi / 2)

            ex = px + vx * self.TIMESTEP
            ey = py + vy * self.TIMESTEP

            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - self.HUMAN_DIAMETER / 2 - self.ROBOT_RADIUS

            if closest_dist < dmin:
                dmin = closest_dist

        for human in self.static_humans + self.dynamic_humans:
            px = human.x - self.robot.x
            py = human.y - self.robot.y

            vx = human.speed * np.cos(human.orientation) - action[0] * np.cos(
                action[2] * self.TIMESTEP + self.robot.orientation) - action[1] * np.cos(
                action[2] * self.TIMESTEP + self.robot.orientation + np.pi / 2)
            vy = human.speed * np.sin(human.orientation) - action[0] * np.sin(
                action[2] * self.TIMESTEP + self.robot.orientation) - action[1] * np.sin(
                action[2] * self.TIMESTEP + self.robot.orientation + np.pi / 2)

            ex = px + vx * self.TIMESTEP
            ey = py + vy * self.TIMESTEP

            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - self.HUMAN_DIAMETER / 2 - self.ROBOT_RADIUS

            if closest_dist < dmin:
                dmin = closest_dist

        for interaction in (self.moving_interactions + self.static_interactions + self.h_l_interactions):
            px = interaction.x - self.robot.x
            py = interaction.y - self.robot.y

            speed = 0
            if interaction.name == "human-human-interaction":
                for h in interaction.humans:
                    speed += h.speed
                speed /= len(interaction.humans)

            vx = speed * np.cos(human.orientation) - action[0] * np.cos(
                action[2] * self.TIMESTEP + self.robot.orientation) - action[1] * np.cos(
                action[2] * self.TIMESTEP + self.robot.orientation + np.pi / 2)
            vy = speed * np.sin(human.orientation) - action[0] * np.sin(
                action[2] * self.TIMESTEP + self.robot.orientation) - action[1] * np.sin(
                action[2] * self.TIMESTEP + self.robot.orientation + np.pi / 2)

            ex = px + vx * self.TIMESTEP
            ey = py + vy * self.TIMESTEP

            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - self.HUMAN_DIAMETER / 2 - self.ROBOT_RADIUS

            if closest_dist < dmin:
                dmin = closest_dist

        info = {
            "OUT_OF_MAP": False,
            "COLLISION_HUMAN": False,
            "COLLISION_OBJECT": False,
            "COLLISION": False,
            "DISCOMFORT_SNGNN": 0.0,
            "DISCOMFORT_DSRNN": 0.0,
            'sngnn_reward': 0.0,
            'distance_reward': 0.0,

            # metrics with different nomenclature
            "SUCCESS": False,
            "COLLISION_WALL": False,
            "TIMEOUT": False,
            "FAILURE_TO_PROGRESS": self.failure_to_progress,
            "STALLED_TIME": self.stalled_time,
            "TIME_TO_REACH_GOAL": None,  # will be None if the robot didn't reach the goal.
            "STL": None,
            "SPL": None
        }

        # calculate the reward and record necessary information
        if self.MAP_X / 2 < self.robot.x or self.robot.x < -self.MAP_X / 2 or self.MAP_Y / 2 < self.robot.y or self.robot.y < -self.MAP_Y / 2:
            self._is_terminated = self._is_terminated or self.END_WITH_COLLISION
            self._collision = True
            info["OUT_OF_MAP"] = True
        # THESIS. in my thesis, i simplly set reach goal and success when distance_to_goal < self.GOAL_THRESHOLD
        # if distance_to_goal < self.GOAL_THRESHOLD and diff_goal_angle < self.GOAL_ORIENTATION_THRESHOLD:
        if distance_to_goal < self.GOAL_THRESHOLD:
            self._is_terminated = True
            info["SUCCESS"] = True
            info["TIME_TO_REACH_GOAL"] = self.ticks * self.TIMESTEP

        if collision is True:
            self._is_terminated = self._is_terminated or self.END_WITH_COLLISION
            self._collision = True
            info["COLLISION"] = True

            if collision_human:
                info["COLLISION_HUMAN"] = True

            if collision_object:
                info["COLLISION_OBJECT"] = True

        if self.ticks >= self.EPISODE_LENGTH:
            self._is_truncated = True
            info["TIMEOUT"] = True

        if collision_wall:
            info["COLLISION_WALL"] = True

        if distance_to_goal > self.prev_goal_distance:
            self.failure_to_progress += 1
            info["FAILURE_TO_PROGRESS"] = self.failure_to_progress
        self.prev_goal_distance = distance_to_goal

        robot_speed = np.linalg.norm(np.array([action[0], action[1]], dtype=np.float32))
        if robot_speed == 0:
            self.stalled_time += self.TIMESTEP
            info["STALLED_TIME"] = self.stalled_time

        self.robot_path_length += self.TIMESTEP * robot_speed
        info["PATH_LENGTH"] = self.robot_path_length

        self.reward_calculator.update_env(self)
        reward = self.reward_calculator.compute_reward(action, self._prev_observations, self._current_observations)
        for k, v in self.reward_calculator.info.items():
            info[k] = v

        # velocity based info
        self.v_min = min(self.v_min, robot_speed)
        self.v_max = max(self.v_max, robot_speed)
        self.v_avg *= (self.ticks - 1)
        self.v_avg += robot_speed
        self.v_avg /= self.ticks

        info["V_MIN"] = self.v_min
        info["V_AVG"] = self.v_avg
        info["V_MAX"] = self.v_max

        # acceleration based info
        vel = np.array([action[0], action[1]], dtype=np.float32)
        a = (vel - self.prev_vel) / self.TIMESTEP
        self.prev_vel = vel
        a_magnitude = np.linalg.norm(a)

        self.a_min = min(self.a_min, a_magnitude)
        self.a_max = max(self.a_max, a_magnitude)
        self.a_avg *= (self.ticks - 1)
        self.a_avg += a_magnitude
        self.a_avg /= self.ticks

        info["A_MIN"] = self.a_min
        info["A_AVG"] = self.a_avg
        info["A_MAX"] = self.a_max

        jerk = (a - self.prev_a) / self.TIMESTEP
        self.prev_a = a
        jerk_magnitude = np.linalg.norm(jerk)
        self.jerk_min = min(self.jerk_min, jerk_magnitude)
        self.jerk_max = max(self.jerk_max, jerk_magnitude)
        self.jerk_avg *= (self.ticks - 1)
        self.jerk_avg += jerk_magnitude
        self.jerk_avg /= self.ticks

        info["JERK_MIN"] = self.jerk_min
        info["JERK_AVG"] = self.jerk_avg
        info["JERK_MAX"] = self.jerk_max

        # calculating the closest distance to humans and time to collision
        closest_human_dist = float('inf')
        time_to_collision = None
        robot_vx = action[0] * np.cos(self.robot.orientation) + action[1] * np.cos(self.robot.orientation + np.pi / 2)
        robot_vy = action[0] * np.sin(self.robot.orientation) + action[1] * np.sin(self.robot.orientation + np.pi / 2)

        for h in self.static_humans + self.dynamic_humans:
            closest_human_dist = min(closest_human_dist, np.sqrt((self.robot.x - h.x) ** 2 + (self.robot.y - h.y) ** 2))
            t = compute_time_to_collision(
                self.robot.x,
                self.robot.y,
                robot_vx,
                robot_vy,
                h.x,
                h.y,
                h.speed * np.cos(h.orientation),
                h.speed * np.sin(h.orientation),
                self.ROBOT_RADIUS,
                self.HUMAN_DIAMETER / 2
            )

            if t != -1:
                if time_to_collision is None:
                    time_to_collision = ceil(t / self.TIMESTEP)
                else:
                    time_to_collision = min(time_to_collision, ceil(t / self.TIMESTEP))

        for i in self.moving_interactions + self.static_interactions:
            for h in i.humans:
                closest_human_dist = min(closest_human_dist,
                                         np.sqrt((self.robot.x - h.x) ** 2 + (self.robot.y - h.y) ** 2))
                t = compute_time_to_collision(
                    self.robot.x,
                    self.robot.y,
                    robot_vx,
                    robot_vy,
                    h.x,
                    h.y,
                    h.speed * np.cos(h.orientation),
                    h.speed * np.sin(h.orientation),
                    self.ROBOT_RADIUS,
                    self.HUMAN_DIAMETER / 2
                )

                if t != -1:
                    if time_to_collision is None:
                        time_to_collision = ceil(t / self.TIMESTEP)
                    else:
                        time_to_collision = min(time_to_collision, ceil(t / self.TIMESTEP))

        for i in self.h_l_interactions:
            closest_human_dist = min(closest_human_dist,
                                     np.sqrt((self.robot.x - i.human.x) ** 2 + (self.robot.y - i.human.y) ** 2))
            h = i.human
            t = compute_time_to_collision(
                self.robot.x,
                self.robot.y,
                robot_vx,
                robot_vy,
                h.x,
                h.y,
                h.speed * np.cos(h.orientation),
                h.speed * np.sin(h.orientation),
                self.ROBOT_RADIUS,
                self.HUMAN_DIAMETER / 2
            )

            if t != -1:
                if time_to_collision is None:
                    time_to_collision = ceil(t / self.TIMESTEP)
                else:
                    time_to_collision = min(time_to_collision, ceil(t / self.TIMESTEP))

        if time_to_collision is None:
            info["TIME_TO_COLLISION"] = -1
        else:
            info["TIME_TO_COLLISION"] = time_to_collision

        info["MINIMUM_DISTANCE_TO_HUMAN"] = closest_human_dist

        if (closest_human_dist - self.ROBOT_RADIUS) >= 0.45:  # same value used in SEAN 2.0
            self.compliant_count += 1

        info["PERSONAL_SPACE_COMPLIANCE"] = self.compliant_count / self.ticks

        # Success weighted by time length
        info["STL"] = 0
        info["SPL"] = 0
        if info["SUCCESS"]:
            if self.has_orca_robot_collided or not self.has_orca_robot_reached_goal:
                info["STL"] = 1
                info["SPL"] = 1
            else:
                stl_value = float(self.orca_robot_reach_time / self.ticks)
                if stl_value > 1: stl_value = 1
                info["STL"] = stl_value

                spl_value = float(self.orca_robot_path_length / info["PATH_LENGTH"])
                if spl_value > 1: spl_value = 1
                info["SPL"] = spl_value

        closest_obstacle_dist = float('inf')
        obstacle_dist_sum = 0
        obstacle_count = 0
        for p in self.plants:
            d = np.sqrt((self.robot.x - p.x) ** 2 + (self.robot.y - p.y) ** 2) - self.PLANT_RADIUS
            closest_obstacle_dist = min(closest_obstacle_dist, d)
            obstacle_dist_sum += d
            obstacle_count += 1

        for table in self.tables:
            p_x, p_y = get_nearest_point_from_rectangle(table.x, table.y, table.length, table.width, table.orientation,
                                                        self.robot.x, self.robot.y)
            d = np.sqrt((self.robot.x - p_x) ** 2 + (self.robot.y - p_y) ** 2)
            closest_obstacle_dist = min(
                closest_obstacle_dist,
                d
            )
            obstacle_dist_sum += d
            obstacle_count += 1

        for chair in self.chairs:
            p_x, p_y = get_nearest_point_from_rectangle(chair.x, chair.y, chair.length, chair.width, chair.orientation,
                                                        self.robot.x, self.robot.y)
            d = np.sqrt((self.robot.x - p_x) ** 2 + (self.robot.y - p_y) ** 2)
            closest_obstacle_dist = min(
                closest_obstacle_dist,
                d
            )
            obstacle_dist_sum += d
            obstacle_count += 1

        for wall in self.walls:
            p_x, p_y = get_nearest_point_from_rectangle(wall.x, wall.y, wall.length, wall.thickness, wall.orientation,
                                                        self.robot.x, self.robot.y)
            d = np.sqrt((self.robot.x - p_x) ** 2 + (self.robot.y - p_y) ** 2)
            closest_obstacle_dist = min(
                closest_obstacle_dist,
                d
            )
            obstacle_dist_sum += d
            obstacle_count += 1

        info["MINIMUM_OBSTACLE_DISTANCE"] = closest_obstacle_dist
        info["AVERAGE_OBSTACLE_DISTANCE"] = None
        if obstacle_count > 0:
            info["AVERAGE_OBSTACLE_DISTANCE"] = obstacle_dist_sum / obstacle_count

        # information of the interacting entities within the environment
        info["interactions"] = {}
        info["interactions"]["human-human"] = []
        info["interactions"]["human-laptop"] = []

        curr_humans = len(self.static_humans + self.dynamic_humans)
        curr_laptops = len(self.laptops)

        for i, interaction in enumerate(self.moving_interactions + self.static_interactions):
            interaction_indices = []
            count_of_humans = 0

            for j, human in enumerate(interaction.humans):
                interaction_indices.append(curr_humans + j)
                count_of_humans += 1

            for p in range(len(interaction_indices)):
                for q in range(p + 1, len(interaction_indices)):
                    info["interactions"]["human-human"].append((interaction_indices[p], interaction_indices[q]))
                    info["interactions"]["human-human"].append((interaction_indices[q], interaction_indices[p]))

            curr_humans += count_of_humans

        for i, interaction in enumerate(self.h_l_interactions):
            info["interactions"]["human-laptop"].append((curr_humans + i, curr_laptops + i))
            # assertion statement
            if (i == len(self.h_l_interactions)):
                assert (curr_humans + i == (self.total_humans - 1))
                assert (curr_laptops + i == len(self.laptops + self.h_l_interactions) - 1)

        return reward, info





