# This is the python class to be used by
# the RL learning algorithm

# This class communicates with the CoppeliaSim env via ROS
# and provides a gym a wrapper of the CoppeliaSim env
# to the RL learning algorithm

# Topics involved
# /sync_gym: synchronization channel from gym env (this class) to coppelia script

import rospy
import std_msgs
from sensor_msgs.msg import Image
import time
import gym
from gym import spaces
import numpy as np

class Manipulation_Env(gym.Env):
    def __init__(self, noise=0.0):
        self.sync_pub = rospy.Publisher('sync_gym', std_msgs.msg.Int32, queue_size = 10)
        self.sync_sub = rospy.Subscriber('sync_sim', std_msgs.msg.Int32, self.state_cb)
        self.image_sub = rospy.Subscriber('camera_image', Image, self.image_cb)
        self.joint_sub = rospy.Subscriber('joint_state', std_msgs.msg.Float32MultiArray, self.joint_cb)
        self.action_pub = rospy.Publisher('action', std_msgs.msg.Float32MultiArray, queue_size = 10)
        self.sync_state = -1 # -1 for uninitialized
        self.sequence = 0 # the current sequence (step) number
        self.noise = noise # Option for noise on the joint observations
        self.joint_sync = 0 # Currently received joint msg seq
        self.camera_sync = 0 # Currently received camera msg seq
        # The observation is ready when both joint_sync and camera_sync are equal to sequence
        self.cur_image = 0
        self.cur_joint = 0

        # Gym observation and action space definition
        # TODO: high/low for spaces
        self.observation_space = spaces.Dict(
            {   # Position/Velocity/Acceleration for 7 joints
                "joint": spaces.Box(-1, 1, shape=(21,), dtype=float), # What's the correct high/low?
                "camera": spaces.Box(0, 1, shape=(256,256,3), dtype=float),
            }
        )

        self.action_space = spaces.Dict(
            {
                "joint": spaces.Box(-1, 1, shape=(7,), dtype=float),
                "camera": spaces.Box(-1, 1, shape=(6,), dtype=float),
            }
        )

        self.start_wait_for_sim()
        if(not self.initialized):
            print("Failed to receive communication from simulator within \
                the designated attempts, environment creation failed")

    def state_cb(self, data):
        self.sync_state = data.data

    # Load the image from the ROS message
    # and update the sync variable
    def image_cb(self, data):
        self.camera_sync = data.header.seq
        self.cur_image = np.asarray(data.data, dtype=float) / 256
        self.cur_image = self.cur_image.reshape((256,256,3))

    # Using the first number of the float32multiarray
    # as the sync number since this message
    # does not have a header
    def joint_cb(self, data):
        self.joint_sync = int(data.data[0])
        self.cur_joint = np.asarray(data.data[1:])

    # Decide on a finish condition
    def step(self, action):
        self.sequence += 1
        # TODO: publish actions via publishers

        attempts = 20
        # Wait until both camera and joint sync are updated
        while((self.camera_sync < self.sequence) or (self.joint_sync < self.sequence)):
            time.sleep(0.1)
            attempts -= 1
            if(attempts < 0):
                print("Failed to receive state reply from simulator in step()")
                # TODO: raise error and fail
        
        # TODO: collect and return state, reward, done, info

    # TODO: either make a new publisher for reset
    # or simply use one of the numbers in action_pub to
    # indicate whether the simulator should reset
    # After publishing the reset message
    def reset(self):
        self.sequence = 1
        self.camera_sync = -1
        self.joint_sync = -1
        attempts = 20
        while((not self.camera_sync == 1) or (not self.joint_sync == 1)):
            time.sleep(0.1)
            attempts -= 1
            if(attempts < 0):
                print("Failed to receive initial state after sending reset message")
                # TODO: raise error and fail
        return

    # publish sync 0 to simulator and wait for a reply
    # when the simulator also publishes 0, both sides are up and ready
    # if the simulator is not up after 20 attempts, return failure
    def start_wait_for_sim(self, attempts = 20):
        self.sync_pub.publish(0)
        while(self.sync_state == -1):
            self.sync_pub.publish(0)
            attempts -= 1
            if(attempts < 0):
                self.initialized = False
                return
            time.sleep(1)
        self.initialized = True