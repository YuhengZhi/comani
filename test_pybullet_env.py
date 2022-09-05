import numpy as np
import cv2
from pybullet_two_link import Manipulation_Env

env = Manipulation_Env()
outVideo = cv2.VideoWriter("test_env_max.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (480,480))
obs = env.reset()
outVideo.write((obs["camera"]*255).astype(np.uint8))
done = False
while(not done):
    obs, reward, done, info = env.step({"joint": [2.0, 0.5], "camera": [0,0,1]})
    outVideo.write((obs["camera"]*255).astype(np.uint8))
print(reward)
outVideo.release()