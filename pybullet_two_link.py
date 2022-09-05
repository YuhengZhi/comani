# This is a python class to be used for a 2D grasping
# experiment. The camera can be moved in two directions

# This class provides a Gym wrapper for a pybullet simulation

import pybullet as p
import time
import pybullet_data
import math
import numpy as np
import gym
from gym import spaces

class Manipulation_Env(gym.Env):
    def __init__(self, noise=0.0):
        self.observation_space = spaces.Dict(
            {   # Position and velocity for 7 joints
                "joint": spaces.Box( # Position and then velocity
                    np.asarray([0, -3, -5, 5], dtype=float),
                    np.asarray([1,3,5,5], dtype=float),
                    shape=(4,), dtype=float
                ),
                "camera": spaces.Box(0, 1, shape=(480,480,3), dtype=float),
            }
        )

        self.action_space = spaces.Dict(
            {   # Velocity for 7 joints
                "joint": spaces.Box(-5, 5, shape=(2,), dtype=float),
                "camera": spaces.Box(np.asarray([-1,-1,1]), np.asarray([1,1,3]), shape=(3,), dtype=float),
            }
        )
        self.noise = noise

        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)

        initialPos = [0,0,0.03]
        initialOrientation = p.getQuaternionFromEuler([0,0,0])
        self.plane = p.loadURDF("plane.urdf")
        self.sphere = p.loadURDF("./urdf/sphere.urdf", globalScaling=0.05)
        self.two_link = p.loadURDF("./urdf/two_link.urdf", initialPos, initialOrientation)
        #self.fixation = p.createConstraint(self.plane, -1, self.two_link, 0, p.JOINT_FIXED, [0,0,1], [0,0,1], [0,0,0])

        self.spherePos = [0.12,0.02,0.005]  # Reset sphere to this position
        self.sphereOrientation = p.getQuaternionFromEuler([0,0,0])

        self.target_pos = np.asarray([-0.1,-0.2])
        self.cur_ep = 0
        self.max_ep = 1000

        # Give color to sphere
        #p.changeVisualShape(self.sphere, 2, rgbaColor=[0.8,0.1,0.1,1.0])

        # Also give color to arm
        p.changeVisualShape(self.two_link, 0, rgbaColor=[0.1,0.8,0.1,1.0])
        p.changeVisualShape(self.two_link, 2, rgbaColor=[0.1,0.1,0.8,1.0])

        self.pixelWidth = 480
        self.pixelHeight = 480
        self.aspect = 1
        self.camDistance = 2

        self.fov = 60 # Default fov
    
    def __del__(self):
        p.disconnect()

    def reset(self):
        # Move sphere to reset location
        p.resetBasePositionAndOrientation(self.sphere, self.spherePos, self.sphereOrientation)
        # Reset joint position and velocity control
        p.resetJointState(self.two_link, 0, 0)
        p.resetJointState(self.two_link, 2, 0)
        p.setJointMotorControlArray(self.two_link, [0,1], p.VELOCITY_CONTROL, targetVelocities=[0]*2, forces=[20,20])
        p.stepSimulation()
        time.sleep(0.5) # Return the initial state
        view = p.computeViewMatrix([0,0,self.camDistance], [0,0,0], [0,1,0])
        projection = p.computeProjectionMatrixFOV(self.fov, self.aspect, 0.5, 5.0)
        _, _, curimg, _, _ = p.getCameraImage(self.pixelWidth, self.pixelHeight, view, projection)
        curimg = np.asarray(curimg, dtype=float)
        self.cur_ep = 0
        return {"joint": np.zeros(2), "camera": np.stack([curimg[:,:,2],curimg[:,:,1],curimg[:,:,0]], axis=2)/255}
    
    def step(self, action):
        p.setJointMotorControlArray(self.two_link, [0,1], p.VELOCITY_CONTROL, targetVelocities=action["joint"], forces=[5,5])
        p.stepSimulation()
        time.sleep(1.0/240)
        view = p.computeViewMatrix([action["camera"][0], action["camera"][1], self.camDistance],
            [action["camera"][0], action["camera"][1], 0], [0,1,0])
        projection = p.computeProjectionMatrixFOV(self.fov / action["camera"][2], self.aspect, 0.5, 5.0)
        _, _, curimg, _, _ = p.getCameraImage(self.pixelWidth, self.pixelHeight, view, projection)
        curimg = np.asarray(curimg, dtype=float)
        cur_joint = p.getJointStates(self.two_link, [0,1])
        cur_joint = [cur_joint[0][0], cur_joint[1][0]]

        sphere_pos = p.getBasePositionAndOrientation(self.sphere)[0][:2]
        self.cur_ep += 1
        done = False
        reward = 0
        if(self.cur_ep >= self.max_ep):
            done = True
        distance = np.sum(np.square(self.target_pos - np.asarray(sphere_pos)))
        if(distance < 0.05):
            reward = 10
            done = True
        return {"joint": np.asarray(cur_joint), "camera": np.stack([curimg[:,:,2],curimg[:,:,1],curimg[:,:,0]], axis=2)/255}, reward, done, "N/A"