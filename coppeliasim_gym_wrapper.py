import sim
import gym
from gym import spaces
import sys
import time
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dm_control.utils import rewards

class CoppeliaSim_Env(gym.Env):
    def __init__(self):

        # Connect to CoppeliaSim
        print ('Program started')
        sim.simxFinish(-1) # Close all open connections
        self.clientID = sim.simxStart('127.0.0.1',19997,True,True,5000,5)

        if self.clientID != -1:
            print ('Connected to remote API server')
        else:
            sys.exit('Connection to remote API server unsuccessful')

        # Set inital variables
        self.target_pos         = np.array([0.6,0.15,0.4])
        self.max_episode_steps  = 1000
        self.ep_counter         = 0
        self.iter_counter       = 0
        self.vid_record_int     = 20

        # Start simulation
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_oneshot)

        # Obtain all object handles
        self.getObjectHandle()

        # Set the initial position of the red sphere
        sim.simxSetObjectPosition(self.clientID, self.sph_Handle, -1, self.target_pos, sim.simx_opmode_oneshot)

        # Initialize camera frame
        _, resolution, image    = sim.simxGetVisionSensorImage(self.clientID, self.cam_Handle, 0, sim.simx_opmode_streaming)
        time.sleep(0.2)

        # Initial observation
        self.img           = np.zeros((3,84,84), dtype = np.uint8)
        # self.img2           = np.zeros((3,84,84), dtype = np.uint8)
        # self.img3           = np.zeros((3,84,84), dtype = np.uint8)
        self.initial_obs    = self.get_obs()

        # Observation Space
        self.observation_space = gym.spaces.Box(0, 255, shape=(84,84,3), dtype=np.uint8)

        # Action Space
        self.action_space   = gym.spaces.Box(-1, 1, shape=(7,), dtype=np.float32)

        # Limits
        self.low            = np.asarray([-1,-1,-1,-1,-1,-1,-1], dtype=np.float32)
        self.high           = np.asarray([1,1,1,1,1,1,1], dtype=np.float32)

    def getObjectHandle(self):

        # Object handle for Panda Base Frame
        self.bf_Handle   = sim.simxGetObjectHandle(self.clientID, "./base_frame", sim.simx_opmode_blocking)[1]

        # Object handle for Panda joints
        self.joint_Handle = [0]*7
        for i in range(7):
            jointPath               = "./Franka_joint" + str(i+1)
            self.joint_Handle[i]    = sim.simxGetObjectHandle(self.clientID, jointPath, sim.simx_opmode_blocking)[1]

        # Object handle for Panda Gripper Center Joint
        self.gripper_Handle   = sim.simxGetObjectHandle(self.clientID, "./Franka/FrankaGripper_centerJoint", sim.simx_opmode_blocking)[1]

        # Object handle for camera
        self.cam_Handle   = sim.simxGetObjectHandle(self.clientID, "./Cam2", sim.simx_opmode_blocking)[1]

        # Object handle for target sphere
        self.sph_Handle   = sim.simxGetObjectHandle(self.clientID, "./Sphere", sim.simx_opmode_blocking)[1]

        # print(self.joint_Handle)

        return (self.joint_Handle)

    def get_obs(self):

        # Build 3 Observations
        # self.img[6:9]    = self.img[3:6]
        # self.img[3:6]    = self.img[0:3]

        # Get camera frame as observation
        _, resolution, image  = sim.simxGetVisionSensorImage(self.clientID, self.cam_Handle, 0, sim.simx_opmode_buffer)

        image    = np.array(image, dtype = np.uint8)
        image.resize([resolution[0], resolution[1], 3])
        image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image    = cv2.flip(image,0)
        image    = image.reshape(3,84,84)
        time.sleep(0.005)

        # self.img[0:3]   = image
        self.img   = image

        # Stack of 3 Observations
        # img     = np.vstack((self.img1, self.img2))
        # img     = np.vstack((img, self.img3))

        return(self.img)

    def set_joint_pos(self, _handle, _pos):

        # Input: Object handle and Joint position -> Joint position is a scalar value
        # Calls the script in CoppeliaSim to change the handle to position mode and write the desired position

        _, _, _, _, _ = sim.simxCallScriptFunction(self.clientID, '/Franka', sim.sim_scripttype_childscript, 'resetAllJoints', [_handle], [_pos], [], '', sim.simx_opmode_blocking)
        time.sleep(0.1)

    def set_joint_vel(self, _handle, _vel):

        # Input: Object handle and Joint velocity -> Joint velocity is a scalar value
        # Sets the handle to desired velocity

        _ = sim.simxSetJointTargetVelocity(self.clientID, _handle, _vel, sim.simx_opmode_streaming)
        time.sleep(0.003)

    def reset(self):

        print('Resetting the environment')

        # Update episode counter
        self.ep_counter         += 1

        # Stop the simulation to reset position and velocity of all joints of the robot and to reset the camera
        self.stop_simulation()

        # Restart the simulation
        self.restart_simulation()

        # Initialize camera frame
        _, resolution, image    = sim.simxGetVisionSensorImage(self.clientID, self.cam_Handle, 0, sim.simx_opmode_streaming)
        time.sleep(0.2)

        # Initialize Gripper handle
        _, self.grip_pos    = sim.simxGetObjectPosition(self.clientID, self.gripper_Handle, -1, sim.simx_opmode_streaming)
        time.sleep(0.05)

        # Initial observation
        self.img            = np.zeros((3,84,84), dtype = np.uint8)
        # self.img2           = np.zeros((3,84,84), dtype = np.uint8)
        # self.img3           = np.zeros((3,84,84), dtype = np.uint8)
        self.initial_obs    = self.get_obs()

        # print('Episode: ', self.ep_counter)
        # print('obs in reset: ', self.initial_obs.shape)

        return(self.initial_obs)

    def step(self, action):

        assert(self.action_space.contains(action))

        # Set the joints to velocities as per the action received
        for i in range(len(self.joint_Handle)):
            self.set_joint_vel(self.joint_Handle[i], action[i])

        # Get griper position
        _, self.grip_pos    = sim.simxGetObjectPosition(self.clientID, self.gripper_Handle, -1, sim.simx_opmode_buffer)
        self.reward         = self.compute_reward(np.asarray(self.grip_pos), self.target_pos)

        # Updated observation and reward computation
        obs                 = self.get_obs()
        info                = {"status": "Goal not reached"}

        # Conditions to terminate an epsiode
        self.iter_counter   += 1
        done                = False
        if(self.iter_counter >= self.max_episode_steps):
            done = True
            for i in range(7):
                self.set_joint_vel(self.joint_Handle[i], 0)

        # Update info dict as per reward value
        if self.reward == np.float32(1) :
            info = {"status": "Goal reached"}

        # info = {"is_success": done, "episode": self.iter_counter}

        # print('Reward: {} and Iteration: {}'.format(self.reward, self.iter_counter))

        return obs, self.reward, done, info

    def compute_reward(self, _achieved_goal, _desired_goal):

        distance    = np.linalg.norm( _achieved_goal - _desired_goal )
        distance    = round(distance, 2)
        #reward      = -1 * distance # This definition is to be used in continuous reward scenario
        # reward      = np.float32(0) # Sparse reward setting

        # Update the reward when end effector has reached the goal position
        # if(distance <= 0.05):
        #     reward  = np.float32(1)
        reward      = rewards.tolerance(distance, bounds = (0, 0.05), margin = 0.05)

        return reward

    def record_video(self):

        # Get updated frame from the camera
        _, resolution, image  = sim.simxGetVisionSensorImage(self.clientID, self.cam_Handle, 0, sim.simx_opmode_buffer)

        # Save Image and write it to VideoWriter
        img = np.array(image, dtype = np.uint8)
        img.resize([resolution[0], resolution[1], 3])
        img = np.flipud(img)
        self.video_output.write(img)
        time.sleep(0.005)

    def restart_simulation(self):

        # Start simulation
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_oneshot)

        # Set the initial position of the red sphere
        sim.simxSetObjectPosition(self.clientID, self.sph_Handle, -1, self.target_pos, sim.simx_opmode_oneshot)

        # Reset iteration counter
        self.iter_counter       = 0

    def stop_simulation(self):

        # Stop the running simulation
        sim.simxStopSimulation(self.clientID, sim.simx_opmode_oneshot)

        # Wait until the updated status is received
        is_running = True
        while is_running:
            error_code, ping_time = sim.simxGetPingTime(self.clientID)
            error_code, server_state = sim.simxGetInMessageInfo(self.clientID, sim.simx_headeroffset_server_state)
            is_running = server_state & 1
