import numpy as np
import cv2
from pybullet_two_link import Manipulation_Env

env = Manipulation_Env()
outVideo = cv2.VideoWriter("test_pybullet_env.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (480,480))
obs = env.reset()
outVideo.write((obs["camera"]*255).astype(np.uint8))
for i in range(200):
    obs, reward, done, info = env.step({"joint": [2.0,0.5], "camera": [0,0,1]})
    outVideo.write((obs["camera"]*255).astype(np.uint8))
outVideo.release()