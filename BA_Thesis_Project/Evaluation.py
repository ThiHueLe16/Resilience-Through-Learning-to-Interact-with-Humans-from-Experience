import sys
import time
import gymnasium as gym
import numpy as np
import yaml
import gymnasium_envCustomHue
import importlib, inspect
import inspect, gymnasium_envCustomHue

from BA_Thesis_Project.SafetyEnvelopeWrapperB1 import SafetyEnvelopeWrapperB1
from BA_Thesis_Project.ppo_agent import PPOAgent


mod = importlib.import_module("gymnasium_envCustomHue.envs.frontal_encounter")


configPath= "../SocNavGym/environment_configs/exp1_with_sngnn.yaml"
custom_eval_config_path= "../gymnasium_envCustomHue/envs/eval_frontal_encounter_configs.yaml"
tensorboard_eval_log_dir="tensorboard_eval_log_dir"
trained_model_B1_path="./save_trained_model/final_trained_architecture_B1"
trained_model_B2_path="./save_trained_model/final_trained_architecture_B2"
trained_model_B3_path="./save_trained_model/final_trained_architecture_B3"

def evaluate_model_B1():
    eval_env=gym.make('gymnasium_envCustomHue/FrontalEncounter-v0', socnavgym_env_config=configPath,custom_scenario_config=custom_eval_config_path)
    eval_env_inside_wrapper=SafetyEnvelopeWrapperB1(eval_env)

    # load custom config path to see how many scenario +name:
    with open(custom_eval_config_path, "r") as custom_eval_yaml:
        scenario_config= yaml.safe_load(custom_eval_yaml)
    # load list of custom scenario
    eval_scenarios= scenario_config["scenario"]["eval_scenarios"]
    number_scenario= len(eval_scenarios)

    # load train model
    model= PPOAgent(env=eval_env_inside_wrapper, tensorboard_log=tensorboard_eval_log_dir)
    model.load(path=trained_model_B1_path)

    episode_metrics=[]
    for i in range(number_scenario):
        # start each defined eval scennario
        eval_scenario_name = eval_scenarios[(i+1)%number_scenario]["id"]
        print(f"running scenario:{eval_scenario_name}")

        # calculate shortest path (for path-efficiency metric)
        obs, info = eval_env_inside_wrapper.reset()
        goal_x_in_robot_frame=obs["robot"][6]
        goal_y_in_robot_frame = obs["robot"][7]
        d_shortest=np.sqrt(goal_x_in_robot_frame**2+goal_y_in_robot_frame**2 )
        last_info={}
        done=False
        while not done:
            # action = eval_env.action_space.sample()
            action,_=model.act(obs)
            obs, reward, terminated, truncated, info = eval_env_inside_wrapper.step(action)
            done=terminated or truncated
            last_info=info
            eval_env.render()
            # time.sleep(0.2)
        eval_env.close()
        safety_intervened = bool(last_info.get("safety_envelope_intervenes", False))
        success = bool(last_info.get("SUCCESS", False))
        path_length= last_info.get("PATH_LENGTH")

        if success and path_length > 0:
            path_efficiency = d_shortest / path_length
        else:
            path_efficiency = 0.0

        if safety_intervened or not success:
            utility = 0.0
        else:
            utility = path_efficiency

        episode_metrics.append({"success": success, "safety_intervened": safety_intervened,"path_efficiency": path_efficiency,"utility": utility})
    success_rate = np.mean([1.0 if m["success"] else 0.0 for m in episode_metrics])
    intervention_rate = np.mean([1.0 if m["safety_intervened"] else 0.0 for m in episode_metrics])
    mean_path_efficiency = np.mean([m["path_efficiency"] for m in episode_metrics if m["success"]])
    mean_utility = np.mean([m["utility"] for m in episode_metrics])

if __name__ == "__main__":
    evaluate_model_B1()

