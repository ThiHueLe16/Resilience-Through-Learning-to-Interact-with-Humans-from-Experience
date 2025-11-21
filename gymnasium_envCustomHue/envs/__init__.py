
# #
from  gymnasium_envCustomHue.envs.frontal_encounter import FrontalEncounter
from gymnasium.envs.registration import register

register(
    id="gymnasium_envCustomHue/FrontalEncounter-v0",
    entry_point="gymnasium_envCustomHue.envs.frontal_encounter:FrontalEncounter",
)
