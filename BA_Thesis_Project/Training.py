from stable_baselines3.common.monitor import Monitor

from BA_Thesis_Project.SafetyEnvelopeWrapperB1 import SafetyEnvelopeWrapperB1
from BA_Thesis_Project.SafetyEnvelopeWrapperB2 import SafetyEnvelopeWrapperB2
from BA_Thesis_Project.SafetyEnvelopeWrapperB3 import SafetyEnvelopeWrapperB3
from BA_Thesis_Project.ppo_agent import PPOAgent
import gymnasium as gym
from stable_baselines3.common.callbacks import (CheckpointCallback, EvalCallback, CallbackList)
import importlib
from pathlib import Path

mod = importlib.import_module("gymnasium_envCustomHue.envs.frontal_encounter")

TOTAL_TRAIN_TIMESTEPS= 100000

ID_CUSTOM_ENV='gymnasium_envCustomHue/FrontalEncounter-v0'
SOCNAVENV_CONFIG_PATH="../SocNavGym/environment_configs/exp1_with_sngnn.yaml"

CUSTOM_TRAIN_CONFIG_PATH="../gymnasium_envCustomHue/envs/train_frontal_encounter_configs.yaml"
# CUSTOM_EVAL_CONFIG_PATH="../gymnasium_envCustomHue/envs/eval_frontal_encounter_configs.yaml"

TENSORBOARD_LOGDIR= "tensorboard_train_log_dir_UPDATE"
# log dir for final model when using save()
SAVE_TRAINED_MODEL_DIR= "save_trained_model_UPDATE"
# Create log dir where evaluation and checkpoint results will be saved
CALLBACK_LOG_DIR= Path("callback_log_dir_UPDATE")

EVAL_FREQ=10000
N_EVAL_EPISODE=20
SAVE_CHECKPOINT_FREQ=20000

def train_architecture_b1():
    # make train env
    env = gym.make(id=ID_CUSTOM_ENV, socnavgym_env_config=SOCNAVENV_CONFIG_PATH, custom_scenario_config=CUSTOM_TRAIN_CONFIG_PATH)
    # wrapped this env around the env by safety env -> we have an env inside the safety envelope of the baseline architecture, which interact directly with the RL-agent
    env_inside_wrapper= SafetyEnvelopeWrapperB1(env)
    # reset the agent to set up the alertness_value, safety_env_intervene in obs
    obs_train, _=env_inside_wrapper.reset()
    # create the RL agent
    rl_agent_baseline= PPOAgent(env=env_inside_wrapper, tensorboard_log=TENSORBOARD_LOGDIR)
    # make eval env
    eval_env= gym.make(id=ID_CUSTOM_ENV, socnavgym_env_config=SOCNAVENV_CONFIG_PATH, custom_scenario_config=CUSTOM_TRAIN_CONFIG_PATH)
    eval_env_inside_wrapper=SafetyEnvelopeWrapperB1(eval_env)
    eval_env =Monitor(eval_env_inside_wrapper)
    obs_eval, _= eval_env.reset()
    # set up callback
    callback_log_dir=CALLBACK_LOG_DIR/"modelB1"
    name_prefix="modelB1"

    checkpoint_callback = CheckpointCallback(save_freq=SAVE_CHECKPOINT_FREQ, save_path=str(callback_log_dir), name_prefix=name_prefix)
    eval_callback=EvalCallback(eval_env, best_model_save_path=str(callback_log_dir),
                              log_path=str(callback_log_dir), eval_freq=EVAL_FREQ,
                              n_eval_episodes=N_EVAL_EPISODE, deterministic=True,
                              render=False)
    callback= CallbackList([checkpoint_callback, eval_callback])

    #train agent
    print("Training B1 architecture started...")
    rl_agent_baseline.train(total_timesteps=TOTAL_TRAIN_TIMESTEPS, tb_log_name="architecture_B1", callback=callback)
    print("Training B1 architecture ended...")
    # save trained model
    rl_agent_baseline.save(path=SAVE_TRAINED_MODEL_DIR + "/final_trained_architecture_B1")
    print(f"Trained B1 model was saved to {SAVE_TRAINED_MODEL_DIR}/final_trained_architecture_B1")

def train_architecture_b2():
    env = gym.make(id=ID_CUSTOM_ENV, socnavgym_env_config=SOCNAVENV_CONFIG_PATH, custom_scenario_config=CUSTOM_TRAIN_CONFIG_PATH)
    env_inside_wrapper= SafetyEnvelopeWrapperB2(env)
    env_inside_wrapper.reset()
    rl_agent_baseline= PPOAgent(env=env_inside_wrapper, tensorboard_log=TENSORBOARD_LOGDIR)

    # make eval env
    eval_env = gym.make(id=ID_CUSTOM_ENV, socnavgym_env_config=SOCNAVENV_CONFIG_PATH,
                        custom_scenario_config=CUSTOM_TRAIN_CONFIG_PATH)
    eval_env_inside_wrapper = SafetyEnvelopeWrapperB2(eval_env)
    eval_env = Monitor(eval_env_inside_wrapper)
    obs_eval, _ = eval_env.reset()
    # set up callback
    callback_log_dir = CALLBACK_LOG_DIR / "modelB2"
    name_prefix = "modelB2"

    checkpoint_callback = CheckpointCallback(save_freq=SAVE_CHECKPOINT_FREQ, save_path=str(callback_log_dir),name_prefix=name_prefix)
    eval_callback = EvalCallback(eval_env, best_model_save_path=str(callback_log_dir),
                                 log_path=str(callback_log_dir), eval_freq=EVAL_FREQ,
                                 n_eval_episodes=N_EVAL_EPISODE, deterministic=True,render=False)
    callback = CallbackList([checkpoint_callback, eval_callback])

    # train agent
    print("Training B2 architecture started...")
    rl_agent_baseline.train(total_timesteps=TOTAL_TRAIN_TIMESTEPS, tb_log_name="architecture_B2",callback=callback)
    print("Training B2 architecture ended...")
    rl_agent_baseline.save(path=SAVE_TRAINED_MODEL_DIR + "/final_trained_architecture_B2")
    print(f"Trained B2 model was saved to {SAVE_TRAINED_MODEL_DIR}/final_trained_architecture_B2")

def train_architecture_b3():

    env = gym.make(id=ID_CUSTOM_ENV, socnavgym_env_config=SOCNAVENV_CONFIG_PATH, custom_scenario_config=CUSTOM_TRAIN_CONFIG_PATH)
    env_inside_wrapper= SafetyEnvelopeWrapperB3(env)
    env_inside_wrapper.reset()
    rl_agent_baseline= PPOAgent(env=env_inside_wrapper, tensorboard_log=TENSORBOARD_LOGDIR)

    # make eval env

    eval_env = gym.make(id=ID_CUSTOM_ENV, socnavgym_env_config=SOCNAVENV_CONFIG_PATH,
                        custom_scenario_config=CUSTOM_TRAIN_CONFIG_PATH)
    eval_env_inside_wrapper = SafetyEnvelopeWrapperB3(eval_env)
    eval_env = Monitor(eval_env_inside_wrapper)
    obs_eval, _ = eval_env.reset()
    # set up callback
    callback_log_dir = CALLBACK_LOG_DIR / "modelB3"
    name_prefix = "modelB3"

    checkpoint_callback = CheckpointCallback(save_freq=SAVE_CHECKPOINT_FREQ, save_path=str(callback_log_dir),
                                             name_prefix=name_prefix)
    eval_callback = EvalCallback(eval_env, best_model_save_path=str(callback_log_dir),
                                 log_path=str(callback_log_dir), eval_freq=EVAL_FREQ,
                                 n_eval_episodes=N_EVAL_EPISODE, deterministic=True,
                                 render=False)
    callback = CallbackList([checkpoint_callback, eval_callback])

    #train agent
    print("Training B3 architecture started...")
    rl_agent_baseline.train(total_timesteps=TOTAL_TRAIN_TIMESTEPS, tb_log_name="architecture_B3", callback=callback)
    print("Training B3 architecture ended...")
    # save trained model
    rl_agent_baseline.save(path=SAVE_TRAINED_MODEL_DIR + "/final_trained_architecture_B3")
    print(f"Trained B3 model was saved to {SAVE_TRAINED_MODEL_DIR}/final_trained_architecture_B3_UPDATE")

def continue_training_architecture_b3(load_path: str,extra_timesteps: int,):

    # 1. Recreate env + wrapper (same as before)
    env = gym.make(id=ID_CUSTOM_ENV,socnavgym_env_config=SOCNAVENV_CONFIG_PATH,custom_scenario_config=CUSTOM_TRAIN_CONFIG_PATH,)
    env_inside_wrapper = SafetyEnvelopeWrapperB3(env)
    env_inside_wrapper.reset()

    # 2. Load existing model into PPOAgent
    #    (adapt this if your PPOAgent has a different load API)

    rl_agent_baseline = PPOAgent(env=env_inside_wrapper,tensorboard_log=TENSORBOARD_LOGDIR)
    rl_agent_baseline.load(path=load_path)

    # 3. Make eval env again
    eval_env = gym.make(
        id=ID_CUSTOM_ENV,
        socnavgym_env_config=SOCNAVENV_CONFIG_PATH,
        custom_scenario_config=CUSTOM_TRAIN_CONFIG_PATH,
    )
    eval_env_inside_wrapper = SafetyEnvelopeWrapperB3(eval_env)
    eval_env = Monitor(eval_env_inside_wrapper)
    obs_eval, _ = eval_env.reset()

    print("RESUME EVAL obs space:", eval_env_inside_wrapper.observation_space)
    print("RESUME EVAL obs shapes:", {k: v.shape for k, v in obs_eval.items()})

    # 4. Reuse SAME callback dirs so logs and best-model continue there
    callback_log_dir = CALLBACK_LOG_DIR / "modelB3"
    name_prefix = "modelB3"

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_CHECKPOINT_FREQ,
        save_path=str(callback_log_dir),
        name_prefix=name_prefix,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(callback_log_dir),
        log_path=str(callback_log_dir),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODE,
        deterministic=True,
        render=False,
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    # 5. Continue training — IMPORTANT: use SAME tb_log_name
    print(f"Continuing B3 training from {load_path} ...")
    rl_agent_baseline.train(
        total_timesteps=extra_timesteps,
        tb_log_name="architecture_B3",   # same as before → same TB run
        callback=callback,
    )
    print("Continuing B3 training ended...")

    # 6. Optionally save a new "final" model
    rl_agent_baseline.save(path=SAVE_TRAINED_MODEL_DIR + "/final_trained_architecture_B3")
    print(f"Continued B3 model was saved to {SAVE_TRAINED_MODEL_DIR}/final_trained_architecture_B3")
def continue_training_architecture_b2(load_path: str,extra_timesteps: int,):
    # Recreate env + wrapper
    env = gym.make(id=ID_CUSTOM_ENV,socnavgym_env_config=SOCNAVENV_CONFIG_PATH,custom_scenario_config=CUSTOM_TRAIN_CONFIG_PATH,)
    env_inside_wrapper = SafetyEnvelopeWrapperB2(env)
    env_inside_wrapper.reset()

    #  Load existing model into PPOAgent
    rl_agent_baseline = PPOAgent(env=env_inside_wrapper,tensorboard_log=TENSORBOARD_LOGDIR)
    rl_agent_baseline.load(path=load_path)

    #  Make eval env again
    eval_env = gym.make(
        id=ID_CUSTOM_ENV,
        socnavgym_env_config=SOCNAVENV_CONFIG_PATH,
        custom_scenario_config=CUSTOM_TRAIN_CONFIG_PATH,
    )
    eval_env_inside_wrapper = SafetyEnvelopeWrapperB2(eval_env)
    eval_env = Monitor(eval_env_inside_wrapper)
    obs_eval, _ = eval_env.reset()

    #  Reuse SAME callback dirs so logs and best-model continue there
    callback_log_dir = CALLBACK_LOG_DIR / "modelB2"
    name_prefix = "modelB2"

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_CHECKPOINT_FREQ,
        save_path=str(callback_log_dir),
        name_prefix=name_prefix,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(callback_log_dir),
        log_path=str(callback_log_dir),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODE,
        deterministic=True,
        render=False,
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    #  Continue training — IMPORTANT: use SAME tb_log_name
    print(f"Continuing B2 training from {load_path} ...")
    rl_agent_baseline.train(
        total_timesteps=extra_timesteps,
        tb_log_name="architecture_B2",   # same as before → same TB run
        callback=callback,
    )
    print("Continuing B2 training ended...")

    # Optionally save a new "final" model
    rl_agent_baseline.save(path=SAVE_TRAINED_MODEL_DIR + "/final_trained_architecture_B2")
    print(f"Continued B2 model was saved to {SAVE_TRAINED_MODEL_DIR}/final_trained_architecture_B2")

def continue_training_architecture_b1(load_path: str,extra_timesteps: int,):

    # Recreate env + wrapper (same as before)
    env = gym.make(id=ID_CUSTOM_ENV,socnavgym_env_config=SOCNAVENV_CONFIG_PATH,custom_scenario_config=CUSTOM_TRAIN_CONFIG_PATH,)
    env_inside_wrapper = SafetyEnvelopeWrapperB1(env)
    env_inside_wrapper.reset()

    #  Load existing model into PPOAgent
    rl_agent_baseline = PPOAgent(env=env_inside_wrapper,tensorboard_log=TENSORBOARD_LOGDIR)
    rl_agent_baseline.load(path=load_path)

    #Make eval env again
    eval_env = gym.make(id=ID_CUSTOM_ENV,socnavgym_env_config=SOCNAVENV_CONFIG_PATH,custom_scenario_config=CUSTOM_TRAIN_CONFIG_PATH,)
    eval_env_inside_wrapper = SafetyEnvelopeWrapperB1(eval_env)
    eval_env = Monitor(eval_env_inside_wrapper)
    obs_eval, _ = eval_env.reset()

    # Reuse SAME callback dirs so logs and best-model continue there
    callback_log_dir = CALLBACK_LOG_DIR / "modelB1"
    name_prefix = "modelB1"

    checkpoint_callback = CheckpointCallback(save_freq=SAVE_CHECKPOINT_FREQ,save_path=str(callback_log_dir),name_prefix=name_prefix,)

    eval_callback = EvalCallback(eval_env,best_model_save_path=str(callback_log_dir),log_path=str(callback_log_dir),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODE,
        deterministic=True,
        render=False,
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    # 5. Continue training — IMPORTANT: use SAME tb_log_name
    print(f"Continuing B1 training from {load_path} ...")
    rl_agent_baseline.train(
        total_timesteps=extra_timesteps,
        tb_log_name="architecture_B1",   # same as before → same TB run
        callback=callback,
    )
    print("Continuing B1 training ended...")

    # 6. Optionally save a new "final" model
    rl_agent_baseline.save(path=SAVE_TRAINED_MODEL_DIR + "/final_trained_architecture_B1")
    print(f"Continued B1 model was saved to {SAVE_TRAINED_MODEL_DIR}/final_trained_architecture_B1")

if __name__ == "__main__":
    # train_architecture_b1()
    #  train_architecture_b2()
    #  train_architecture_b3()
    # continue_training_architecture_b1("./save_trained_model_UPDATE/final_trained_architecture_B1.zip", 100000)
    continue_training_architecture_b2("./save_trained_model_UPDATE/final_trained_architecture_B2.zip", 100000)
    # continue_training_architecture_b3("./save_trained_model_UPDATE/final_trained_architecture_B3.zip", 100000)

