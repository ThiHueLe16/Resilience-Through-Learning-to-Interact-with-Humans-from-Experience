import time

import gymnasium as gym
from stable_baselines3 import PPO
from torch.backends.cudnn import deterministic

import gymnasium_envCustomHue


# # Make Python load the local edited version first
# repo_path = "/Users/thihuele/HumanDetection/navgym/SocNavGym"
# if repo_path not in sys.path:
#     sys.path.insert(0, repo_path)

#
#

import importlib, inspect

from BA_Thesis_Project.SafetyEnvelope import TIME_STEP
from BA_Thesis_Project.SafetyEnvelopeWrapperB1 import SafetyEnvelopeWrapperB1
from BA_Thesis_Project.SafetyEnvelopeWrapperB1 import SafetyEnvelopeWrapperB1
from BA_Thesis_Project.SafetyEnvelopeWrapperB2 import SafetyEnvelopeWrapperB2
from BA_Thesis_Project.SafetyEnvelopeWrapperB3 import SafetyEnvelopeWrapperB3

mod = importlib.import_module("gymnasium_envCustomHue.envs.frontal_encounter")
# print("module file:", inspect.getfile(mod))
# print("has FrontalEncounter?", hasattr(mod, "FrontalEncounter"))

# from gymnasium.envs.registration import registry
# print([spec.id for spec in registry.values() if "Hue" in spec.id])


configPath= "./SocNavGym/environment_configs/exp1_with_sngnn.yaml"
custom_train_config_path= "./gymnasium_envCustomHue/envs/train_frontal_encounter_configs.yaml"
# env = gym.make('gymnasium_envCustomHue/FrontalEncounter-v0', config=configPath)
env=gym.make('gymnasium_envCustomHue/FrontalEncounter-v0', socnavgym_env_config=configPath,custom_scenario_config=custom_train_config_path)

env=SafetyEnvelopeWrapperB1(env)
num_episodes = 50
max_steps = 1000 # you can increase if episodes are short
# yhis is to point the env in reward to the wrapper env after using gym.make

# end
model= PPO.load("./BA_Thesis_Project/callback_log_dir_UPDATE/modelB1/best_model.zip",print_system_info=True)

for ep in range(num_episodes):
    print(f"\n--- Episode {ep+1} ---")
    obs, info = env.reset()   # start a new episode (random world)
    base = env.unwrapped
    rwd = getattr(base, "reward_calculator", None)

    if rwd is not None:
        rwd.env = env


    done = False
    step = 0
    total_reward = 0

    # print(f"initiial safety signal when episode begin {base.alertness_value} from episode{ep}")

    while not done and step < max_steps:
        # Take a random action just for testing
        # action = env.action_space.sample()
        action,_=model.predict(obs,deterministic=True)
        # if hasattr(env, "__dict__"):
        #     base.alertness_value = 178980 + ep +step
        #     print(f"hello im using this safety signal {base.alertness_value}")
        # else:
        #     print("Env uses __slots__, cannot dynamically add attributes.")
        # Perform the step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # print(info)
        # print(f"hello episode {ep} at step {step} safety envelope intervenes {base.safety_envelope_intervenes}")

        env.render()
        done = terminated or truncated

        # # # Slow down visualization a bit
        # time.sleep(0.3)

        step += 1

    print(f"Episode {ep+1} finished after {step} steps, total reward = {total_reward:.2f}")

env.close()
print(f"✅ Finished {num_episodes} episodes.")

#====================== ====================== ====================== ====================== ====================== ====================== ======================
# # test eval config
#
# import sys
# import time
#
# import gymnasium as gym
# import yaml
#
# import gymnasium_envCustomHue
#
#
# import importlib, inspect
# import inspect, gymnasium_envCustomHue
# mod = importlib.import_module("gymnasium_envCustomHue.envs.frontal_encounter")
# # print("module file:", inspect.getfile(mod))
# # print("has FrontalEncounter?", hasattr(mod, "FrontalEncounter"))
#
# # from gymnasium.envs.registration import registry
# # print([spec.id for spec in registry.values() if "Hue" in spec.id])
#
#
# configPath= "./SocNavGym/environment_configs/exp1_with_sngnn.yaml"
# custom_eval_config_path= "./gymnasium_envCustomHue/envs/eval_frontal_encounter_configs.yaml"
# # env = gym.make('gymnasium_envCustomHue/FrontalEncounter-v0', config=configPath)
# # env=gym.make('gymnasium_envCustomHue/FrontalEncounter-v0', socnavgym_env_config=configPath,custom_scenario_config=custom_eval_config_path)
#
# # num_episodes = 20
# max_steps = 3 # you can increase if episodes are short
# # yhis is to point the env in reward to the wrapper env after using gym.make
#
# # end
# def run_episode( scenario_index: int, seed:int ,render: bool = False):
#     """Run *one* eval episode for one model on one fixed scenario."""
#     env = gym.make( 'gymnasium_envCustomHue/FrontalEncounter-v0',socnavgym_env_config=configPath, custom_scenario_config=custom_eval_config_path, scenario_index=scenario_index)
#
#     obs, info = env.reset(seed=seed)
#     done = False
#     truncated = False
#     total_reward = 0.0
#
#     while not (done or truncated):
#         # deterministic=True for reproducible evaluation
#         #
#         action = env.action_space.sample()
#         obs, reward, done, truncated, info = env.step(action)
#         total_reward += reward
#
#         if render:
#             env.render()
#         #
#         time.sleep(0.2)
#
#     env.close()
#     return total_reward, info
#
#
#
# #load custom config path to see how many scenario +name:
# with open(custom_eval_config_path, "r") as custom_eval_yaml:
#     scenario_config= yaml.safe_load(custom_eval_yaml)
#
# # load list of custom scenario
# eval_scenarios= scenario_config["scenario"]["eval_scenarios"]
#
# for scen_idx, scen in enumerate(eval_scenarios):
#     scen_name = scen["name"]
#     print(f"\n=== Scenario {scen_idx}: {scen_name} ===")
#
#     print("run model A")
#     total_reward, last_info = run_episode(scen_idx, scen_idx, render=True)
#     print("run model B")
#     total_rewardB, last_infoB = run_episode(scen_idx, scen_idx, render=True)
#
#
# print("\n=== Summary ===")
#

#====================== ====================== ====================== ====================== ====================== ====================== ======================

# test eval config

import sys
import time

# import gymnasium as gym
# import yaml
#
# import gymnasium_envCustomHue
# import importlib, inspect
# import inspect, gymnasium_envCustomHue
#
# from BA_Thesis_Project.SafetyEnvelopeWrapperB3 import SafetyEnvelopeWrapperB3
#
# mod = importlib.import_module("gymnasium_envCustomHue.envs.frontal_encounter")
# # print("module file:", inspect.getfile(mod))
# # print("has FrontalEncounter?", hasattr(mod, "FrontalEncounter"))
#
# # from gymnasium.envs.registration import registry
# # print([spec.id for spec in registry.values() if "Hue" in spec.id])
#
#
# configPath= "./SocNavGym/environment_configs/exp1_with_sngnn.yaml"
# custom_eval_config_path= "./gymnasium_envCustomHue/envs/eval_frontal_encounter_configs.yaml"
# # env = gym.make('gymnasium_envCustomHue/FrontalEncounter-v0', config=configPath)
# env=gym.make('gymnasium_envCustomHue/FrontalEncounter-v0', socnavgym_env_config=configPath,custom_scenario_config=custom_eval_config_path)
# env=SafetyEnvelopeWrapperB3(env)
# # load custom config path to see how many scenario +name:
# with open(custom_eval_config_path, "r") as custom_eval_yaml:
#     scenario_config= yaml.safe_load(custom_eval_yaml)
#
# # load list of custom scenario
# eval_scenarios= scenario_config["scenario"]["eval_scenarios"]
# num_scen= len(eval_scenarios)
# for i in range(num_scen):
#     eval_scen_name=eval_scenarios[(i+1)%num_scen]["id"]
#     print(f"running scenario:{eval_scen_name}")
#     obs, info = env.reset()
#
#     done = False
#     truncated = False
#     total_reward = 0.0
#
#     while not (done or truncated):
#         # deterministic=True for reproducible evaluation
#         #
#         action = env.action_space.sample()
#         obs, reward, done, truncated, info = env.step(action)
#         total_reward += reward
#
#
#         env.render()
#         #
#         # time.sleep(0.2)
#
#     env.close()
