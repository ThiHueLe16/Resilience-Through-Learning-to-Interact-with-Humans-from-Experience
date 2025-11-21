from gymnasium.envs.registration import register

register(
    id="gymnasium_envCustomHue/GridWorld-v0",
    entry_point="gymnasium_envCustomHue.envs:GridWorldEnv",
)
