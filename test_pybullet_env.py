import numpy as np
import cv2
from pybullet_env import Manipulation_Env

env = Manipulation_Env()
outVideo = cv2.VideoWriter("test_env_max.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (84,84))
obs = env.reset()
outVideo.write((obs.transpose([1,2,0])*255).astype(np.uint8))
done = False
total_reward = 0
step_count = 0
while(not done):
    step_count += 1
    obs, reward, done, info = env.step(np.asarray([0.5,0.5,0,0,-1]))
    outVideo.write((obs.transpose([1,2,0])*255).astype(np.uint8))
    total_reward += reward
print(reward)
outVideo.release()