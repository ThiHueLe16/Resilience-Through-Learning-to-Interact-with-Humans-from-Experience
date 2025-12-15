import sys
import time
import csv
import gymnasium as gym
import numpy as np
import yaml
import gymnasium_envCustomHue
import importlib, inspect
import os


from BA_Thesis_Project.SafetyEnvelopeWrapperB1 import SafetyEnvelopeWrapperB1
from BA_Thesis_Project.SafetyEnvelopeWrapperB2 import SafetyEnvelopeWrapperB2
from BA_Thesis_Project.SafetyEnvelopeWrapperB3 import SafetyEnvelopeWrapperB3
from BA_Thesis_Project.ppo_agent import PPOAgent

# make sure custom env us imported
mod = importlib.import_module("gymnasium_envCustomHue.envs.frontal_encounter")


configPath= "../SocNavGym/environment_configs/exp1_with_sngnn.yaml"
custom_eval_config_path= "../gymnasium_envCustomHue/envs/eval_frontal_encounter_configs.yaml"
tensorboard_eval_log_dir_thesis="tensorboard_eval_log_dir_thesis"
eval_result_log_path_csv="eval_results/"

trained_model_B1_path="./callback_log_dir_UPDATE/modelB1/best_model.zip"
trained_model_B2_path="./callback_log_dir_UPDATE/modelB2/best_model.zip"
trained_model_B3_path="./callback_log_dir_UPDATE/modelB3/best_model.zip"


LENGTH_MAX=50
TIME_MAX=500
MAX_VELOCITY=0.1
D_MIN_HUMAN=0.535
D_WARN_HUMAN=1.605

W1_PATH_SCORE = 0.25
W2_LENGTH_SCORE = 0.25
W3_AVG_VEL_SCORE = 0.25
W4_CLOSEST_HUMAN_DIST_SCORE = 0.25




def evaluate_model(architecture_name:str, wrapper_name, model_path:str, csv_output_path:str, render:bool=False):
    """

    :param architecture_name: "B1", "B2", "B3"
    :param wrapper_name: SafetyEnvelopeWrapperB1/SafetyEnvelopeWrapperB2/SafetyEnvelopeWrapperB3
    :param model_path: path to trained (best) model
    :param csv_output_path: str, where to save the episode metric table
    :param render:
    :return:
    """
    # create eval env +wrapper
    eval_env=gym.make('gymnasium_envCustomHue/FrontalEncounter-v0', socnavgym_env_config=configPath,custom_scenario_config=custom_eval_config_path)
    time_step = eval_env.get_wrapper_attr("TIMESTEP")
    a_max_robot = eval_env.get_wrapper_attr("MAX_ADVANCE_ROBOT")
    eval_env_inside_wrapper = wrapper_name(eval_env)

    # load eval scenarios to see how many scenario +name:
    with open(custom_eval_config_path, "r") as custom_eval_yaml:
        scenario_config= yaml.safe_load(custom_eval_yaml)
    # load list of custom scenario
    eval_scenarios= scenario_config["scenario"]["eval_scenarios"]
    number_scenario= len(eval_scenarios)
    # load train model
    model = PPOAgent(env=eval_env_inside_wrapper, tensorboard_log=tensorboard_eval_log_dir_thesis)
    model.load(path=model_path)
    episode_metrics = []

    # loop over all scenarios
    for i in range(number_scenario):
        minimum_distance_human = float("inf")
        # start each defined eval scennario
        # (i+1)%number_scenario] NOT i%number_scenario because is called once u use (i+1)%number_scenario
        # so this reset will count from the scenario 2 , not from scenario 1. the last scenario to be scen. id-0
        eval_scenario_name = eval_scenarios[(i+1)%number_scenario]["id"]
        print(f"running scenario:{eval_scenario_name}")


        obs, info = eval_env_inside_wrapper.reset()

        last_info={}
        step_count=0
        done=False
        while not done:
            # action = eval_env.action_space.sample()
            action,_=model.act(obs)
            obs, reward, terminated, truncated, info = eval_env_inside_wrapper.step(action)
            done=terminated or truncated
            # update episode-wide minimum distance to human
            d=info.get("MINIMUM_DISTANCE_TO_HUMAN", None)
            if d is not None:
                minimum_distance_human = min(float(d), minimum_distance_human)
            last_info=info
            step_count+=1
            if render:
                eval_env.render()
        safety_intervened = bool(last_info.get("safety_envelope_intervenes", False))
        success = bool(last_info.get("SUCCESS", False))
        timeout= bool(last_info.get("TIMEOUT", False))
        path_length = last_info.get("PATH_LENGTH")
        average_velocity = last_info.get("V_AVG")
        path_length = float(path_length) if path_length is not None else None
        average_velocity = float(average_velocity) if average_velocity is not None else 0.0
        episode_duration= step_count*time_step

        if success:
            # time score (smaller time -> higher score)
            time_score= 1.0 - np.clip(episode_duration / TIME_MAX, 0.0, 1.0)
            # path length score (smaller length -> higher score)
            if path_length is None or path_length <= 0:
                path_length_score = 0.0
            else:
                path_length_score = 1.0 - float(np.clip(float(path_length) / LENGTH_MAX, 0.0, 1.0))
            avg_vel_score=average_velocity/MAX_VELOCITY

            if minimum_distance_human == float("inf"):
                closest_human_distance_score = 0.0
            else:
                closest_human_distance_score = (float(minimum_distance_human) - D_MIN_HUMAN) / (D_WARN_HUMAN - D_MIN_HUMAN)
                closest_human_distance_score = float(np.clip(closest_human_distance_score, 0.0, 1.0))
        else:
            time_score=0
            path_length_score = 0
            avg_vel_score = 0
            closest_human_distance_score =0


        if safety_intervened or not success:
            utility = 0.0
        else:
            utility = W1_PATH_SCORE*path_length_score+ W2_LENGTH_SCORE*time_score+ W3_AVG_VEL_SCORE*avg_vel_score+W4_CLOSEST_HUMAN_DIST_SCORE*closest_human_distance_score
            utility = float(np.clip(utility, 0.0, 1.0))


        episode_metrics.append({
            "architecture": architecture_name,
            "scenario_id":eval_scenario_name,
            "success": success,
            "safety_intervened": safety_intervened,
            "time_out":timeout,
            "time_score":time_score,
            "path_length_score":path_length_score,
            "avg_vel_score":avg_vel_score,
            "closest_human_distance_score":closest_human_distance_score,
            "episode_duration": episode_duration,
            "path_length":path_length,
            "min_distance_to_human":minimum_distance_human,
            "average_velocity":average_velocity,
            "utility": utility})
    eval_env.close()
    success_rate = np.mean([1.0 if m["success"] else 0.0 for m in episode_metrics])
    intervention_rate = np.mean([1.0 if m["safety_intervened"] else 0.0 for m in episode_metrics])
    mean_path_length = np.mean([m["path_length"] for m in episode_metrics if m["success"]])
    mean_episode_duration=np.mean([m["episode_duration"] for m in episode_metrics if m["success"]])
    mean_min_distance_to_human=np.mean([m["min_distance_to_human"] for m in episode_metrics if m["success"]])
    mean_average_velocity=np.mean([m["average_velocity"] for m in episode_metrics if m["success"]])
    mean_utility = np.mean([m["utility"] for m in episode_metrics])

    print(f"\n=== Summary for {architecture_name} ===")
    print(f"Success rate:         {success_rate:.3f}")
    print(f"Intervention rate:    {intervention_rate:.3f}")
    print(f"Mean path length: {mean_path_length:.3f}")
    print(f"Mean_episode_duration: {mean_episode_duration:.3f}")
    print(f"Mean_average_velocity: {mean_average_velocity:.3f}")
    print(f"Mean_min_distance_to_human: {mean_min_distance_to_human:.3f}")
    print(f"Mean utility:         {mean_utility:.3f}")
    print(f"Results saved to:     {csv_output_path}\n")

    # ensure directory exist before writing
    dir_name = os.path.dirname(csv_output_path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)
    #  Write per-episode metrics to CSV
    if len(episode_metrics) > 0:
        fieldnames = list(episode_metrics[0].keys())
        with open(csv_output_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_metrics)

    summary_csv_path = csv_output_path.replace(".csv", "_summary.csv")
    summary_row = {
        "architecture": architecture_name,
        "num_episodes": len(episode_metrics),
        "success_rate": float(success_rate),
        "intervention_rate": float(intervention_rate),
        "mean_path_length": float(mean_path_length) if np.isfinite(mean_path_length) else "",
        "mean_episode_duration": float(mean_episode_duration) if np.isfinite(mean_episode_duration) else "",
        "mean_average_velocity": float(mean_average_velocity) if np.isfinite(mean_average_velocity) else "",
        "mean_min_distance_to_human": float(mean_min_distance_to_human) if np.isfinite(mean_min_distance_to_human) else "",
        "mean_utility": float(mean_utility),
    }

    # ensure directory exists (reuse your existing dir creation)
    dir_name = os.path.dirname(summary_csv_path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)

    with open(summary_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    print(f"Summary saved to:     {summary_csv_path}")

    return episode_metrics



if __name__ == "__main__":
    # evaluate_model_B1()
    evaluate_model(
        architecture_name="B1",
        wrapper_name=SafetyEnvelopeWrapperB1,
        model_path=trained_model_B1_path,
        csv_output_path=eval_result_log_path_csv+"eval_results_B1.csv",
        render=True,
    )

    # B2
    evaluate_model(
        architecture_name="B2",
        wrapper_name=SafetyEnvelopeWrapperB2,
        model_path=trained_model_B2_path,
        csv_output_path=eval_result_log_path_csv+"eval_results_B2.csv",
        render=True,
    )

    # B3 (proposed architecture)
    evaluate_model(
        architecture_name="B3",
        wrapper_name=SafetyEnvelopeWrapperB3,
        model_path=trained_model_B3_path,
        csv_output_path=eval_result_log_path_csv+"eval_results_B3.csv",
        render=True,
    )

