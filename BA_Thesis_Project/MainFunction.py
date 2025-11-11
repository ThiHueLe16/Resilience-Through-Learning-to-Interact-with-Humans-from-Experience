import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from BA_Thesis_Project.SafetyEnvelope import SafetyEnvelope

CONFIG_PATH="./SocNavGym/environment_configs/exp1_with_sngnn.yaml"
A_ROBOT_MAX_BRAKE=0.5
TIME_STEP=1
V_HUMAN=2
A_MIN_HUMAN_BRAKE =0.2
MAX_ADVANCE_ROBOT=0.1


class MainFunction:
    def __init__(self):
        self.min_distance_to_obstacles=float('inf')


    def make_env(self):
        return gym.make("SocNavGym-v1", config=CONFIG_PATH)

    def run(self,n_episodes=10, sleep_s=0.02, deterministic=True):
        env= self.make_env()
        # get the inner base env. Important so that later we can set the alertness value to the base of env where reward class read to compute the reward.
        # Reward class read value from the base env to compute reward. when using gym.make() to create env, it auto create a wrapper outside. write value using
        # env.alertness_value only set value to the wrapper, not the inner base env
        base= env.unwrapped

        safetyEnvelope= SafetyEnvelope(env,A_ROBOT_MAX_BRAKE,TIME_STEP, V_HUMAN,A_MIN_HUMAN_BRAKE,MAX_ADVANCE_ROBOT )
        # TODO:set this model later after training. for now set it here as an example
        model = PPO.load("./test_hue_saves/ppo_continuous/final_ppo_exp1_no_sngnn.zip",  print_system_info=True)

        for ep in range(n_episodes):
            obs, _ = env.reset()
            terminated = truncated = False
            ep_reward = 0.0
            self.min_distance_to_obstacles= MainFunction.compute_initial_d_min_to_obstacles(obs)

            print(f'episode {ep + 1} begin')
            while not (terminated or truncated):
                main_function_action, _ = model.predict(obs, deterministic=deterministic)
                next_action, base.alertness_value = safetyEnvelope.next_action(self.min_distance_to_obstacles, main_function_action, obs)
                if base.alertness_value==1:
                    terminated=True
                    base.safety_envelope_intervenes=True

                obs, reward, terminated, truncated, info = env.step(next_action)

                self.min_distance_to_obstacles=info['MINIMUM_OBSTACLE_DISTANCE']
                # print(info)
                # print("/////////////////////////////////////////////////////////////////////////")
                ep_reward += reward
                env.render()  # update the window
                time.sleep(sleep_s)  # slow it down (adjust for speed)
            print(f"Episode {ep + 1}: reward={ep_reward:.2f}")

        env.close()
        return

    @staticmethod
    def compute_initial_d_min_to_obstacles(obs):
        """
            Return the initial minimum distance to all others obstacles. Need only for the begining, after 1 first step,
            this distance is returned in the info after executing env.step(action)
        """
        d_min = float('inf')
        # extract robot radius from the observation
        robot_radius = obs["robot"][8]

        # predict new distance from robot to each obstacle and find the minimum
        for key in ["humans", "plants", "laptops", "tables", "walls"]:
            obstacle_obs = obs[key]
            assert (len(obstacle_obs % 14 == 0))
            number_of_obstacles = int(len(obstacle_obs) / 14)
            for i in range(0, number_of_obstacles):
                obstacle_relative_position_x = obstacle_obs[6 + i * 14]
                obstacle_relative_position_y = obstacle_obs[7 + i * 14]
                obstacle_radius = obstacle_obs[10 + i * 14]
                distance= np.sqrt(obstacle_relative_position_x**2 + obstacle_relative_position_y**2 ) -robot_radius -obstacle_radius
                if d_min > distance:
                    d_min =distance
        return d_min
