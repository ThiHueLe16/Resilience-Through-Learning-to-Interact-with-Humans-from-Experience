import numpy as np

TIME_STEP=1
A_ROBOT_MAX_BRAKE=1
V_HUMAN=0.1
A_MIN_HUMAN_BRAKE=1
MAX_ADVANCE_ROBOT=0.1

EPSILON=MAX_ADVANCE_ROBOT
WARNING_ZONE_HUMAN=4
WARNING_ZONE_WALL=2


class SafetyEnvelope:
    # TODO: check away to get timestep from the config file in other class before passing and create the safety envelope object
    # todo also check a_max_human and robot, robot

    def __init__(self,env,a_robot_max_brake=A_ROBOT_MAX_BRAKE, time_step=TIME_STEP,v_human=V_HUMAN,  a_min_human_brake=A_MIN_HUMAN_BRAKE, max_advance_robot=MAX_ADVANCE_ROBOT):
        self.env=env
        self.time_step= time_step
        self.a_robot_max_brake=a_robot_max_brake
        self.v_human =v_human
        self.a_min_human_brake= a_min_human_brake
        self.max_advance_robot=max_advance_robot


    def next_action(self, main_function_action, obs):
        # """
        # decide the next action to execute by checking if an intervention is needed. Additionally, output an alertness value
        # """
        """

        :param min_distance_to_obstacles: current min distance to other obstacles in env
        :param main_function_action: the predict action by the RL agent
        :param obs: current obs
        :return: safe action, alertness value, should_intervene: bool- if the safety envelope should intervene
        """
        should_intervene, alertness_value = self.should_intervene( main_function_action, obs)
        if should_intervene:
            stop_action = np.array([0, 0, 0], dtype=np.float32)
            return stop_action, alertness_value, should_intervene

        return main_function_action, alertness_value, should_intervene

    # check if an intervention is needed and output additional alertness value
    def should_intervene(self, main_function_action, obs):
        # minimum_obstacles_distance = min_distance_to_obstacles
        # minimum_braking_distance = self.compute_minimum_braking_distance(main_function_action, obs)
        # print(f"minimum braking distance {minimum_braking_distance}")
        # alertness_value = self.compute_continuous_safety_signal(minimum_obstacles_distance,minimum_braking_distance,4 * minimum_braking_distance)
        # return minimum_obstacles_distance <= minimum_braking_distance, alertness_value

        #compute distances to other obstacles
        d_human_current= SafetyEnvelope.compute_d_min_to_obstacles(obs)
        d_walls_current= self.compute_min_distance_to_wall(obs)
        #compute threshold
        d_min_human=self.compute_minimum_braking_distance(main_function_action, obs, "dynamic")
        d_min_wall=self.compute_minimum_braking_distance(main_function_action, obs, "static")
        print(f"d_min human {d_min_human}")
        print(f"d_min wall {d_min_wall}")

        #intervention rules
        intervene_human= d_human_current<=d_min_human
        intervene_wall= d_walls_current<=d_min_wall
        should_intervene=intervene_human or intervene_wall
        #compute alertness_value
        alert_human= self.compute_continuous_safety_signal(d_human_current, d_min_human, WARNING_ZONE_HUMAN*d_min_human)
        alert_wall= self.compute_continuous_safety_signal(d_walls_current, d_min_wall, WARNING_ZONE_WALL*d_min_wall)
        alertness_value= max(alert_wall, alert_human)
        return should_intervene, alertness_value




    # # decide the next action to execute by checking if an intervention is needed. Additionally, output an alertness value
    # def next_action(self,info, main_function_action, obs):
    #     should_intervene, alertness_value=self.should_intervene(info, main_function_action,obs)
    #     if should_intervene:
    #         stop_action = np.array([0, 0, 0], dtype=np.float32)
    #         return stop_action, alertness_value
    #
    #     return main_function_action, alertness_value
    #
    #
    # # check if an intervention is needed and output additional alertness value
    # def should_intervene(self, info,main_function_action,obs):
    #     minimum_obstacles_distance = info['MINIMUM_OBSTACLE_DISTANCE']
    #     minimum_braking_distance=self.compute_minimum_braking_distance(main_function_action,obs)
    #     alertness_value= self.compute_continuous_safety_signal(minimum_obstacles_distance, minimum_braking_distance,2*minimum_braking_distance)
    #     return minimum_obstacles_distance<= minimum_braking_distance,alertness_value



    @staticmethod
    def compute_continuous_safety_signal(d_current, d_min, d_warn):
        """
        :param d_current: current distance of robot to other obstacles
        :param d_min: the min. distance in which robot can still have enough time to accelerate and then try max deceleration until fully stop
        :param d_warn: distance in which robot start to give warning about near intervention
        :return: the alertness value
        """
        print(f"d current={d_current}")
        if d_min >= d_current:
            alertness_value = 1
        elif d_warn > d_current > d_min:
            alertness_value = (d_warn - d_current) / (d_warn - d_min)
        else:
            alertness_value = 0
        return alertness_value


    # this function return the minimum braking distance in which the Safety Envelope need to intervene
    def compute_minimum_braking_distance(self,main_function_action, obs, obstacle_type ):
        """
        :param main_function_action:
        :param obs:
        :param obstacle_type: "dynamic" or "static". use dynamic if obstacle is moving(e.g dynamic human,...) and static (e.g: walls,...) otherwise
        :return: the minimum braking distance in which the Safety Envelope need to intervene
        """
        robot_current_speed= self.get_robot_current_speed()
        print(f"a_robot current speed{robot_current_speed}")
        robot_predicted_speed=SafetyEnvelope.get_action_speed(main_function_action, self.max_advance_robot)
        # robot_radius= obs["robot"][8]
        if obstacle_type=="dynamic":
            # human_radius=obs["humans"][10]
            d_min= (((robot_current_speed+robot_predicted_speed)/2)*self.time_step+ (robot_predicted_speed**2)/(2*self.a_robot_max_brake)+
                ((self.v_human+ self.v_human)/2)*self.time_step+(self.v_human**2)/(2*self.a_min_human_brake))+EPSILON
        else:
            d_min=((robot_current_speed+robot_predicted_speed)/2)*self.time_step+ (robot_predicted_speed**2)/(2*self.a_robot_max_brake)+EPSILON
        return d_min


    def get_robot_current_speed(self):
        """
        :return: current speed of robot
        """
        robot = self.env.get_wrapper_attr("robot")  # returns env.robot if it exists
        robot_v_x, robot_v_y = robot.vel_x, robot.vel_y
        robot_speed = np.sqrt(robot_v_x ** 2 + robot_v_y ** 2)
        return robot_speed
    

    @staticmethod
    def get_action_speed(main_function_action, max_advance_robot):
        """
        :param main_function_action:
        :param max_advance_robot:
        :return:  the speed the action about to take
        """
        # the action space outputs values between -1 and 1 for each action dimension->need to scale those normalized
        # value to the true robot physical limits
        robot_predicted_v_x, robot_predicted_v_y = main_function_action[0] * max_advance_robot, main_function_action[1] * max_advance_robot
        robot_predicted_speed = np.sqrt(robot_predicted_v_x ** 2 + robot_predicted_v_y ** 2)
        print(f"robot action speed ({robot_predicted_v_x,robot_predicted_v_y})")
        return robot_predicted_speed

    def compute_min_distance_to_wall(self,obs):
        # fix error key walls do not exist in obs because of setting padding to True to prevent failure of reshape from SB3
        robot_radius = obs["robot"][8]
        robot_x_world_frame = self.env.get_wrapper_attr("robot").x
        robot_y_world_frame = self.env.get_wrapper_attr("robot").y
        MAP_Y = self.env.get_wrapper_attr("MAP_Y")
        MARGIN = self.env.get_wrapper_attr("MARGIN")
        MAP_X = self.env.get_wrapper_attr("MAP_X")
        HALF_SIZE_X = MAP_X / 2 - MARGIN
        HALF_SIZE_Y = MAP_Y / 2 - MARGIN

        # Distance to each wall (subtract robot radius)
        d_left = (robot_x_world_frame - (-HALF_SIZE_X)) - robot_radius  # distance to left wall
        d_right = (HALF_SIZE_X - robot_x_world_frame) - robot_radius  # distance to right wall
        d_bottom = (robot_y_world_frame - (-HALF_SIZE_Y)) - robot_radius  # distance to bottom wall
        d_top = (HALF_SIZE_Y - robot_y_world_frame) - robot_radius  # distance to top wall
        # The minimum distance to any wall
        return min(d_left, d_right, d_bottom, d_top)


    @staticmethod
    def compute_d_min_to_obstacles( obs):
        """
            Return the initial minimum distance to all others obstacles (NOT TO WALLS, bc of setting padding to prevent error of SB3). Need only for the begining, after 1 first step,
            this distance is returned in the info after executing env.step(action)

        """
        d_min = float('inf')
        # extract robot radius from the observation
        robot_radius = obs["robot"][8]

        # predict new distance from robot to each obstacle and find the minimum, walls is not in keys bc of setting padding to True(prevent error from SB3)
        # for key in ["humans", "plants", "laptops", "tables", "walls"]:
        for key in ["humans", "plants", "laptops", "tables"]:
            if key not in obs:
                continue
            obstacle_obs = obs[key]
            if obstacle_obs.size == 0:
                continue
            assert len(obstacle_obs) % 11 == 0
            number_of_obstacles = int(len(obstacle_obs) / 11)
            for i in range(0, number_of_obstacles):
                obstacle_relative_position_x = obstacle_obs[6 + i * 11]
                obstacle_relative_position_y = obstacle_obs[7 + i * 11]
                obstacle_radius = obstacle_obs[10 + i * 11]
                distance= np.sqrt(obstacle_relative_position_x**2 + obstacle_relative_position_y**2 ) -robot_radius -obstacle_radius
                if d_min > distance:
                    d_min =distance
        return d_min


