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
                    np.asarray([-2.96, -1.83, -2.96, -3.14, -2.96, -0.08, -2.96, -5, -5, -5, -5, -5, -5, -5]),
                    np.asarray([2.96, 1.83, 2.96, 0, 2.96, 3.82, 2.96, 5, 5, 5, 5, 5, 5, 5]), shape=(14,), dtype=float),
                "camera": spaces.Box(0, 1, shape=(1920,1080,3), dtype=float),
            }
        )

        self.action_space = spaces.Dict(
            {   # Velocity for 7 joints
                "joint": spaces.Box(-5, 5, shape=(7,), dtype=float),
                "camera": spaces.Box(-1, 1, shape=(2,), dtype=float),
            }
        )
        self.noise = noise

        self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)
        self.planeId = p.loadURDF("plane.urdf")
        startPos = [0.0, 0.0, 0.1]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.pandaId = p.loadURDF("franka_panda/panda.urdf", startPos, startOrientation)
        self.baseConstraint = p.createConstraint(self.planeId, -1, self.pandaId, 0, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])

        pixelWidth = 1920
        pixelHeight = 1080
        aspect = 1920.0/1080
        camDistance = 4
        self.projection = p.computeProjectionMatrixFOV(60, aspect, 0.5, 5.0)
    
    def __del__(self):
        p.removeConstraint(self.baseConstraint)
        p.disconnect()

    def reset(self):
        # Reset joint position and velocity control
        for i in range(7):
            p.resetJointState(self.pandaId, i, 0)
        p.setJointMotorControlArray(self.pandaId, [0,1,2,3,4,5,6], p.VELOCITY_CONTROL, targetVelocities=[0]*7)
        p.stepSimulation()
        time.sleep(0.5) # Return the initial state
        view = p.computeViewMatrix([0,0,4], [0,0,0], [0,1,0])
        _, _, curimg, _, _ = p.getCameraImage(1920, 1080, view, self.projection)
        return {"joint": np.zeros(14), "camera": np.stack([curimg[:,:,2],curimg[:,:,1],curimg[:,:,0]], axis=2)}
    
    def step(self, action):
        p.setJointMotorControlArray(self.pandaId, [0,1,2,3,4,5,6], p.VELOCITY_CONTROL, action["joint"])
        p.stepSimulation()
        view = p.computeViewMatrix([action["camera"][0], action["camera"][1], 4],
            [action["camera"][0], action["camera"][1], 0], [0,1,0])
        _, _, curimg, _, _ = p.getCameraImage(1920, 1080, view, self.projection)
        cur_joint = p.getJointStates(self.pandaId, [0,1,2,3,4,5,6])
        return {"joint": np.asarray(cur_joint[0]+cur_joint[1]), "camera": np.stack([curimg[:,:,2],curimg[:,:,1],curimg[:,:,0]], axis=2)}, 0, False, "N/A"