# This is a python class to be used for a 2D grasping
# experiment. The camera can be moved in two directions
# This class provides a Gym wrapper for a pybullet simulation
# This version has a seven link arm

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
from collections import deque

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
        self.link_arm = p.loadURDF(directory + "/urdf/seven_link.urdf", initialPos, initialOrientation)
        self.target_sphere = p.loadURDF(directory + '/urdf/sphere_green.urdf', globalScaling=0.05, useFixedBase=True)
        #self.fixation = p.createConstraint(self.plane, -1, self.two_link, 0, p.JOINT_FIXED, [0,0,1], [0,0,1], [0,0,0])

        # Set in the ball_position function
        self.sphere_orientation = p.getQuaternionFromEuler([0,0,0])
        self.radius_min = 0.15
        self.radius_max = 0.32
        self.angle_zone = 20 # Zone for preventing easy hits, in degrees
        self.radius_zone_min = 0.28 # Radius for zone for preventing easy hits
        self.angle_min = 7 # Minimum angle overall
        self.target_distance = 0.04  # Threshold distance for detecting a hit

        self.cur_ep = 0
        self.max_ep = 1000

        # Joint numbers in the simulator
        self.arm_joints = [0, 1, 3, 5, 7, 9, 11]
        self.fingertip_code = 13 # The link ID of the fingertip of the robot
        self.joint_actions = len(self.arm_joints)

        # Also give color to arm
        # Give a special color to the fingertip link
        for i in range(11):
            p.changeVisualShape(self.link_arm, i, rgbaColor=[0.1,0.1,0.8,1.0])
        p.changeVisualShape(self.link_arm, 12, rgbaColor=[0.8,0.1,0.1,1.0])
        p.changeVisualShape(self.link_arm, 13, rgbaColor=[0.8,0.1,0.1,1.0])

        # Observation space is a 3x84x84 image
        # Action space has the controls for the 7 connections
        # and then the camera location and zoom
        self.observation_space = spaces.Box(0, 255, shape=(3,84,84), dtype=np.uint8)
        self.action_space = spaces.Box(-1, 1, shape=(10,), dtype=np.float32)
        # Actual low and high action values
        # needed due to DrQv2 assuming -1 to 1 action space
        self.low = np.asarray([-5] * 7 + [-0.7, -0.7, 1.8], dtype=float)
        self.high = np.asarray([5] * 7 + [0.7, 0.7, 4.5], dtype=float)

        self.pixelWidth = 84
        self.pixelHeight = 84
        self.aspect = 1
        self.camDistance = 2

        self.stack_num = 3
        self.frame_stack = deque([], maxlen=self.stack_num)

        self.fov = 60 # Default fov
    
    def __del__(self):
        p.unloadPlugin(self.plugin)
        p.disconnect()

    def reset(self):
        self.handle_target_reset()

        # Reset joint position and velocity control
        for joint_number in self.arm_joints:
            p.resetJointState(self.link_arm, joint_number, 0)
        p.setJointMotorControlArray(self.link_arm, self.arm_joints, p.VELOCITY_CONTROL, targetVelocities = [0] * 7, forces = [80] * 7)
        p.stepSimulation()
        time.sleep(0.5) # Return the initial state
        view = p.computeViewMatrix([0.7,0.7,self.camDistance], [0.7,0.7,0], [0,1,0])
        projection = p.computeProjectionMatrixFOV(self.fov / 1.8, self.aspect, 0.5, 5.0)
        _, _, curimg, _, _ = p.getCameraImage(self.pixelWidth, self.pixelHeight, view, projection)
        curimg = np.asarray(curimg, dtype=np.uint8).transpose(2,0,1)[:3]
        for i in range(self.stack_num):
            self.frame_stack.append(curimg)
        self.cur_ep = 0

        obs = np.concatenate(list(self.frame_stack), axis=0)
        return obs
    
    # Translate action from -1 to 1 to actual values
    def translate(self, action):
        full_action = {}
        full_action["joint"] = self.low[:self.joint_actions] + (self.high[:self.joint_actions]\
            - self.low[:self.joint_actions]) * (action[:self.joint_actions] + 1) / 2
        full_action["camera"] = self.low[self.joint_actions:] + (self.high[self.joint_actions:]\
            - self.low[self.joint_actions:]) * (action[self.joint_actions:] + 1) / 2
        return full_action
    
    def one_step(self, action):
        assert(self.action_space.contains(action))
        action = self.translate(action)

        # Ask pybullet to set the joints to the indicated velocities        
        p.setJointMotorControlArray(self.link_arm, self.arm_joints, p.VELOCITY_CONTROL, targetVelocities = action["joint"], forces = [80] * 7)
        p.stepSimulation()
        time.sleep(1.0/240)
        view = p.computeViewMatrix([action["camera"][0], action["camera"][1], self.camDistance],
            [action["camera"][0], action["camera"][1], 0], [0,1,0])
        projection = p.computeProjectionMatrixFOV(self.fov / action["camera"][2], self.aspect, 0.5, 5.0)
        _, _, curimg, _, _ = p.getCameraImage(self.pixelWidth, self.pixelHeight, view, projection)
        curimg = np.asarray(curimg, dtype=np.uint8).transpose(2,0,1)[:3]
        self.frame_stack.append(curimg)
        cur_joint = p.getJointStates(self.link_arm, [0,1])
        cur_joint = [cur_joint[0][0], cur_joint[1][0]]

        # Calculate the distance from the fingertip link to the target sphere
        tip_position = np.asarray(p.getLinkState(self.link_arm, self.fingertip_code)[0][:2])
        self.cur_ep += 1
        done = False
        distance = np.sum(np.square(self.target_pos[:2] - tip_position))
        distance = np.sqrt(distance)
        # Some basic reward shaping
        # Return a small reward for being closer to the target
        if(self.cur_ep >= self.max_ep):
            done = True
        if(distance < self.target_distance):
            reward = 1
        else:
            reward = -distance
        # print(str(distance) + '  ' + str(cur_joint) + '  ' + str(self.cur_ep))
        obs = np.concatenate(list(self.frame_stack), axis=0)
        return obs, reward, done, ""
    
    def handle_target_reset(self):
        self.target_pos = np.zeros(3)
        flag = True  # Generate a random target position
        while(flag):  # and go again if the position is in the easy zone
            random_val = np.random.rand(2)
            angle = self.angle_min + random_val[0] * (360 - 2 * self.angle_min)
            radius = self.radius_min + random_val[1] * (self.radius_max - self.radius_min)
            if((angle > self.angle_zone)
                or (angle < 360 - self.angle_zone)
                or (radius < self.radius_zone_min)):
                flag = False
        self.target_pos[0] = radius * np.cos(angle / 180 * np.pi)
        self.target_pos[1] = radius * np.sin(angle / 180 * np.pi)
        self.target_pos[2] = 0.08
        p.resetBasePositionAndOrientation(self.target_sphere, self.target_pos, self.sphere_orientation)
    
    def step(self, action):
        # Step twice for two-step return
        obs, reward, done, _ = self.one_step(action)
        if(done):
            return obs, reward, done, ""
        first_reward = reward
        obs, reward, done, _ = self.one_step(action)
        return obs, first_reward + reward, done, ""
