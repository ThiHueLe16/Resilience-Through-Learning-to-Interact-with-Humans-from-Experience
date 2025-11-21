from BA_Thesis_Project.SafetyEnvelopeWrapper import SafetyEnvelopeWrapper
from BA_Thesis_Project.ppo_agent import PPOAgent
import gymnasium as gym

TOTAL_TRAIN_TIMESTEPS= 1000000

ID_CUSTOM_ENV='gymnasium_envCustomHue/FrontalEncounter-v0'
SOCNAVENV_CONFIG_PATH="../SocNavGym/environment_configs/exp1_with_sngnn.yaml"
CUSTOM_TRAIN_CONFIG_PATH="../gymnasium_envCustomHue/envs/train_frontal_encounter_configs.yaml"

TENSORBOARD_LOGDIR= " ./tensorboard_log_dir"
TB_LOG_NAME_BASELINE_ARCH= "baseline_architecture"
TB_LOG_NAME_PROPOSED_ARCH= "proposed_architecture"

SAVE_TRAINED_MODEL_DIR= "./save_trained_model"



def train_baseline_architecture():
    # TODO ADD CALLBACK CHECKCALL FOR EVALUATION during training
    env = gym.make(id=ID_CUSTOM_ENV, socnavgym_env_config=SOCNAVENV_CONFIG_PATH, custom_scenario_config=CUSTOM_TRAIN_CONFIG_PATH)
    # wrapped this env around the env by safety env -> we have an env inside the safety envelope of the baseline architecture, which interact directly with the RL-agent
    env_inside_baseline_wrapper= SafetyEnvelopeWrapper(env)
    # reset the agent to set up the alertness_value, safety_env_intervene in obs
    env_inside_baseline_wrapper.reset()
    # create the RL agent
    rl_agent_baseline= PPOAgent(env=env_inside_baseline_wrapper, tensorboard_log=TENSORBOARD_LOGDIR)
    #train agent
    print("Training BASELINE architecture started...")
    rl_agent_baseline.train(total_timesteps=TOTAL_TRAIN_TIMESTEPS, tb_log_name=TB_LOG_NAME_BASELINE_ARCH)
    print("Training BASELINE architecture ended...")
    # save trained model
    rl_agent_baseline.save(path=SAVE_TRAINED_MODEL_DIR + "final_trained_baseline_model")
    print(f"Trained BASELINE model was saved to {SAVE_TRAINED_MODEL_DIR}/final_trained_baseline_model")

def train_proposed_architecture():
    # TODO ADD CALLBACK CHECKCALL FOR EVALUATION during training
    env = gym.make(id=ID_CUSTOM_ENV, socnavgym_env_config=SOCNAVENV_CONFIG_PATH,custom_scenario_config=CUSTOM_TRAIN_CONFIG_PATH)
    env_inside_proposed_wrapper = SafetyEnvelopeWrapper(env=env, expose_alertness_value_in_obs=True,shape_reward_with_alertness_value=True)
    env_inside_proposed_wrapper.reset()
    rl_agent_proposed_architecture = PPOAgent(env=env_inside_proposed_wrapper, tensorboard_log=TENSORBOARD_LOGDIR)
    print("Training PROPOSED architecture started...")
    rl_agent_proposed_architecture.train(total_timesteps=TOTAL_TRAIN_TIMESTEPS, tb_log_name=TB_LOG_NAME_PROPOSED_ARCH)
    print("Training PROPOSED architecture ended...")
    # save trained model
    rl_agent_proposed_architecture.save(path=SAVE_TRAINED_MODEL_DIR + "final_trained_proposed_model")
    print(f"Trained PROPOSED model was saved to {SAVE_TRAINED_MODEL_DIR}/final_trained_proposed_model")








