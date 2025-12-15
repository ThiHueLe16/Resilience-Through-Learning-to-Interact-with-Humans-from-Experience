import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

from BA_Thesis_Project.SafetyEnvelopeWrapperB1 import SafetyEnvelopeWrapperB1
import importlib
from pathlib import Path

from BA_Thesis_Project.SafetyEnvelopeWrapperB3 import SafetyEnvelopeWrapperB3

mod = importlib.import_module("gymnasium_envCustomHue.envs.frontal_encounter")
# Import your wrapper
ID_CUSTOM_ENV='gymnasium_envCustomHue/FrontalEncounter-v0'
SOCNAVENV_CONFIG_PATH="../SocNavGym/environment_configs/exp1_with_sngnn.yaml"

CUSTOM_TRAIN_CONFIG_PATH="../gymnasium_envCustomHue/envs/train_frontal_encounter_configs.yaml"
CUSTOM_EVAL_CONFIG_PATH="../gymnasium_envCustomHue/envs/eval_frontal_encounter_configs.yaml"
  # <-- change to your file + class

# Make base env
env = gym.make(id=ID_CUSTOM_ENV, socnavgym_env_config=SOCNAVENV_CONFIG_PATH,
               custom_scenario_config=CUSTOM_TRAIN_CONFIG_PATH)
# Optionally wrap it with your filtering wrapper
# from testRemoveKeyObs import TestFilterWrapper
# env = TestFilterWrapper(env)

# Wrap with your SafetyEnvelope wrapper
env = SafetyEnvelopeWrapperB3(env)
print("\n=== BOUNDS HUMANS ===")
space = env.observation_space.spaces["humans"]
print("low[7] =", space.low[7])
print("high[7] =", space.high[7])
# env = gym.make("SocNavGym-v1", config="../SocNavGym/environment_configs/exp1_with_sngnn.yaml", render_mode="None")
print("\n=== Running Gym Environment Checker ===")
check_env(env, warn=True, skip_render_check=True)
print("=== CHECK COMPLETED ===")
