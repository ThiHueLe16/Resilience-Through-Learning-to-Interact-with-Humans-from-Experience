# THis below path is to make sure that the edited code in socnavenv_v1.py will get run, otherwise it will auto use another file in library not locally
import sys, os
# Make Python load your edited version first
repo_path = "/Users/thihuele/HumanDetection/navgym/SocNavGym"
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

import inspect
import socnavgym
print("🔧 Using socnavgym from:", inspect.getfile(socnavgym), flush=True)
# END
import socnavgym
import gymnasium as gym
from pathlib import Path
# env = gym.make("SocNavGym-v1", config="configTestHue.yaml")

# configPath= "./SocNavGym/environment_configs/static_environments/exp2_staticObstacle_withCorridors.yaml"
configPath= "./SocNavGym/environment_configs/exp1_with_sngnn.yaml"
env = gym.make("SocNavGym-v1", config=configPath )


obs, _ = env.reset()

max_ad_robot_speed= env.get_wrapper_attr("env")
print(f" max_advance_robot{max_ad_robot_speed}")



for i in range(1000):
    print(f'i: {i}')
    robot = env.get_wrapper_attr("robot")  # returns raw_env.robot if it exists
    # print("before")
    # print(f"robot_x, y {robot.x}, {robot.y}")
    # print(f"vx: {robot.vel_x:.3f}, vy: {robot.vel_y:.3f}, va: {robot.vel_a:.3f}, orientation: {robot.orientation}")
    action = env.action_space.sample()
    # print(f"action {action[0]}, {action[1]}, {action[2]}")
    obs, reward, terminated, truncated, info = env.step(action)
    # print("after")
    # ✅ print robot velocities (linear vx, vy, angular omega)


    robot = env.get_wrapper_attr("robot")  # returns raw_env.robot if it exists
    # print(f"robot_x, y {robot.x}, {robot.y}")
    # print(f"vx: {robot.vel_x:.3f}, vy: {robot.vel_y:.3f}, va: {robot.vel_a:.3f}, orientation: {robot.orientation}")
    # print(f"info: {info}")
    # print(f"minimum_distance: {info['MINIMUM_OBSTACLE_DISTANCE']}")
    print(obs)
    goalx, goaly=obs["humans"][6]
    # humanobs=obs["humans"]
    # print(f"humansizevec {len(humanobs)}")
    # print(f"human obs {humanobs}")
    # humanx=obs["humans"][6]
    # humany=obs["humans"][7]
    # print(f"human,x {humanx} human.y {humany}")



    env.render()
    if terminated or truncated:
        env.reset()