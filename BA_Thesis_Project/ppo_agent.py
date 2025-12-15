from stable_baselines3 import PPO


class PPOAgent:
    def __init__(self, env, tensorboard_log:str):
        self.env = env
        # use MultiInputPolicy (not MlpPolicy) because the observation's type of socnavenv is gymnasium.Spaces.Dict
        self.model= PPO("MultiInputPolicy", env=env, verbose=1,tensorboard_log=tensorboard_log)

    def train(self, total_timesteps: int, tb_log_name, callback, reset_Num_Timesteps=False):
            self.model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name, callback=callback,reset_num_timesteps=reset_Num_Timesteps)

    def act(self, obs, deterministic: bool=True):
        """
        given current observation (from env), return an action
        """
        action, _states = self.model.predict(obs, deterministic=deterministic)
        return action, _states

    def save(self, path:str):
        self.model.save(path)

    def load(self, path):
        """
            Load the model from a zip file
        """
        # Load the model from a zip-file. Warning: load re-creates the model from scratch, it does not update it in-place.
        self.model= PPO.load(path=path, env=self.env)

