import numpy as np

A_ROBOT_MAX_BRAKE=2
TIME_STEP=1
V_HUMAN=2
A_MIN_HUMAN_BRAKE=2
MAX_ADVANCE_ROBOT=0.1


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


    def next_action(self, min_distance_to_obstacles, main_function_action, obs):
        # """
        # decide the next action to execute by checking if an intervention is needed. Additionally, output an alertness value
        # """
        """

        :param min_distance_to_obstacles: current min distance to other obstacles in env
        :param main_function_action: the predict action by the RL agent
        :param obs: current obs
        :return: safe action, alertness value, should_intervene: bool- if the safety envelope should intervene
        """
        should_intervene, alertness_value = self.should_intervene(min_distance_to_obstacles, main_function_action, obs)
        if should_intervene:
            stop_action = np.array([0, 0, 0], dtype=np.float32)
            return stop_action, alertness_value, should_intervene

        return main_function_action, alertness_value, should_intervene

    # check if an intervention is needed and output additional alertness value
    def should_intervene(self, min_distance_to_obstacles, main_function_action, obs):
        minimum_obstacles_distance = min_distance_to_obstacles
        minimum_braking_distance = self.compute_minimum_braking_distance(main_function_action, obs)
        alertness_value = self.compute_continuous_safety_signal(minimum_obstacles_distance,minimum_braking_distance,2 * minimum_braking_distance)
        return minimum_obstacles_distance <= minimum_braking_distance, alertness_value


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
        :param d_current:
        :param d_min:
        :param d_warn:
        :return: the alertness value
        """
        if d_min >= d_current:
            alertness_value = 1
        elif d_warn > d_current > d_min:
            alertness_value = (d_warn - d_current) / (d_warn - d_min)
        else:
            alertness_value = 0
        return alertness_value


    # this function return the minimum braking distance in which the Safety Envelope need to intervene
    def compute_minimum_braking_distance(self,main_function_action, obs ):
        """
        :param main_function_action:
        :param obs:
        :return: the minimum braking distance in which the Safety Envelope need to intervene
        """
        robot_current_speed= self.get_robot_current_speed()
        robot_predicted_speed=SafetyEnvelope.get_action_speed(main_function_action, self.max_advance_robot)
        robot_radius= obs["robot"][8]
        human_radius=obs["humans"][10]
        d_min= (((robot_current_speed+robot_predicted_speed)/2)*self.time_step+ (robot_predicted_speed**2)/(2*self.a_robot_max_brake)+
                ((self.v_human+ self.v_human)/2)*self.time_step+(self.v_human**2)/(2*self.a_min_human_brake))-robot_radius-human_radius
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
        return robot_predicted_speed


