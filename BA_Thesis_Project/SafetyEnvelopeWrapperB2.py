
import gymnasium as gym

from BA_Thesis_Project.SafetyEnvelopeWrapperB1 import SafetyEnvelopeWrapperB1

ALERTNESS_VALUE_PENALTY_SCALE= -1000

class SafetyEnvelopeWrapperB2(SafetyEnvelopeWrapperB1):
    def __init__(self, env:gym.Env, alertness_penalty_scale:float=ALERTNESS_VALUE_PENALTY_SCALE):
        super().__init__(env)
        # safe penalty scaler to re-compute the reward later in function step()
        self.alertness_penalty_scale = alertness_penalty_scale


    def step(self, action):
        obs, reward, terminated, truncated, info= super().step(action)
        # update reward function with alertness value
        print(f"reward original {reward}")
        print(f"alertness_value {self.alertness_value}")
        reward += self.alertness_penalty_scale * self.alertness_value
        print(f"new reward{reward}")
        return obs, reward, terminated, truncated, info





