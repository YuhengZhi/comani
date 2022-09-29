# This is a python class to be used for a 2D grasping
# experiment. The camera can be moved in two directions

# This class provides a Gym wrapper for a pybullet simulation

import pybullet as p
import time
import pybullet_data
import math
import numpy as np
import pkgutil
import os
import random
import gym
from gym import spaces

# TODO: Consider adding an action repeat wrapper
class Manipulation_Env(gym.Env):
    def __init__(self, noise=0.0, record=False):
        self.noise = noise
        self.record = record

        self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)
        egl = pkgutil.get_loader('eglRenderer')
        self.plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        initialPos = [0,0,0.03]
        initialOrientation = p.getQuaternionFromEuler([0,0,0])
        directory = os.path.realpath(__file__).split('/')
        directory = '/'.join(directory[:-1])
        self.plane = p.loadURDF("plane.urdf")
        self.sphere = p.loadURDF(directory + "/urdf/sphere.urdf", globalScaling=0.05)
        self.two_link = p.loadURDF(directory + "/urdf/two_link.urdf", initialPos, initialOrientation)
        #self.fixation = p.createConstraint(self.plane, -1, self.two_link, 0, p.JOINT_FIXED, [0,0,1], [0,0,1], [0,0,0])

        # Set in the ball_position function
        self.sphere_pos = []  # Reset sphere to this position
        self.sphere_orientation = p.getQuaternionFromEuler([0,0,0])

        # Target position for the sphere
        self.target_pos = np.asarray([-0.07,0.155])
        self.target_reached = 0.05
        self.cur_ep = 0
        self.max_ep = 300

        # Reward information
        self.success_reward = 10
        self.reward_shaping = 4 # Max reward to assign for reward shaping

        # Give color to sphere
        # Not necessary at the moment due to color being in the urdf
        #p.changeVisualShape(self.sphere, 2, rgbaColor=[0.8,0.1,0.1,1.0])

        # Also give color to arm
        p.changeVisualShape(self.two_link, 0, rgbaColor=[0.1,0.8,0.1,1.0])
        p.changeVisualShape(self.two_link, 2, rgbaColor=[0.1,0.1,0.8,1.0])

        self.observation_space = spaces.Box(0, 1, shape=(3,84,84), dtype=float)
        self.action_space = spaces.Box(-1, 1, shape=(5,), dtype=float)
        # Actual low and high action values
        # needed due to DrQv2 assuming -1 to 1 action space
        self.low = np.asarray([-5,-5,-0.7,-0.7,1], dtype=float)
        self.high = np.asarray([5,5,0.7,0.7,2], dtype=float)

        self.pixelWidth = 84
        self.pixelHeight = 84
        self.aspect = 1
        self.camDistance = 2

        self.fov = 60 # Default fov
    
    def __del__(self):
        p.unloadPlugin(self.plugin)
        p.disconnect()
    
    def ball_position(self):
        draw = random.random()
        if(draw > 0.5):
            # This was previously [0.12, 0.02, 0.005]
            self.sphere_pos = [0.12,0.04,0.005]
            self.target_pos = np.asarray([-0.07, 0.155])
        else:
            self.sphere_pos = [0.12,-0.04,0.005]
            self.target_pos = np.asarray([-0.07, -0.155])

    def reset(self):
        self.ball_position()
        # Move sphere to reset location
        p.resetBasePositionAndOrientation(self.sphere, self.sphere_pos, self.sphere_orientation)
        # Reset joint position and velocity control
        p.resetJointState(self.two_link, 0, 0)
        p.resetJointState(self.two_link, 1, 0)
        p.setJointMotorControlArray(self.two_link, [0,1], p.VELOCITY_CONTROL, targetVelocities=[0]*2, forces=[20,20])
        p.stepSimulation()
        time.sleep(0.5) # Return the initial state
        view = p.computeViewMatrix([0,0,self.camDistance], [0,0,0], [0,1,0])
        projection = p.computeProjectionMatrixFOV(self.fov, self.aspect, 0.5, 5.0)
        _, _, curimg, _, _ = p.getCameraImage(self.pixelWidth, self.pixelHeight, view, projection)
        curimg = np.asarray(curimg, dtype=np.float32) / 255
        self.cur_ep = 0
        return np.stack([curimg[:,:,2], curimg[:,:,1], curimg[:,:,0]], axis=0)
    
    def translate(self, action):
        full_action = {}
        joint_acts = 2
        full_action["joint"] = self.low[:joint_acts] + (self.high[:joint_acts]\
            - self.low[:joint_acts]) * (action[:joint_acts] + 1) / 2
        full_action["camera"] = self.low[joint_acts:] + (self.high[joint_acts:]\
            - self.low[joint_acts:]) * (action[joint_acts:] + 1) / 2
        return full_action
    
    def step(self, action):
        assert(self.action_space.contains(action))
        action = self.translate(action)

        # Ask pybullet to set the joints to the indicated velocities        
        p.setJointMotorControlArray(self.two_link, [0,1], p.VELOCITY_CONTROL, targetVelocities=action["joint"], forces=[5,5])
        p.stepSimulation()
        time.sleep(1.0/240)
        # Changed to fixed camera position to facilitate better training
        #view = p.computeViewMatrix([action["camera"][0], action["camera"][1], self.camDistance],
        #    [action["camera"][0], action["camera"][1], 0], [0,1,0])
        #projection = p.computeProjectionMatrixFOV(self.fov / action["camera"][2], self.aspect, 0.5, 5.0)
        view = p.computeViewMatrix([0, 0, self.camDistance],
            [0, 0, 0], [0,1,0])
        projection = p.computeProjectionMatrixFOV(self.fov / 3.5, self.aspect, 0.5, 5.0)
        _, _, curimg, _, _ = p.getCameraImage(self.pixelWidth, self.pixelHeight, view, projection)
        curimg = np.asarray(curimg, dtype=np.float32) / 255
        cur_joint = p.getJointStates(self.two_link, [0,1])
        cur_joint = [cur_joint[0][0], cur_joint[1][0]]

        # Ask pybullet for the position of the sphere
        sphere_pos = p.getBasePositionAndOrientation(self.sphere)[0][:2]
        self.cur_ep += 1
        done = False
        distance = np.sum(np.square(self.target_pos - np.asarray(sphere_pos)))
        distance = np.sqrt(distance)
        reward = -distance * 0.1
        # Some basic reward shaping
        # Return a small reward for being closer to the target
        if(self.cur_ep >= self.max_ep):
            done = True
        if(distance < self.target_reached):
            reward = 10
            done = True
        # print(str(distance) + '  ' + str(cur_joint) + '  ' + str(self.cur_ep))
        return np.stack([curimg[:,:,2], curimg[:,:,1], curimg[:,:,0]], axis=0), reward, done, ""