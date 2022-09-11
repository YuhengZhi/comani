import torch
import os
import numpy as np
import cv2
from dm_env import specs
from dm_env import TimeStep, StepType
from replay_buffer import ReplayBufferStorage, make_replay_loader
from pybullet_env import Manipulation_Env
from utils import eval_mode
from drqv2 import DrQV2Agent

def assert_exists(directory):
    if(not os.path.exists(directory)):
        os.makedir(directory)

# Configuration variables
num_train_frames = 3100000 # Taken from the medium difficulty rating
num_train_frames = 10000
action_repeat = 2 # Not 100% sure what this means yet. Probably related to multi-step return

record_every = 50 # Record a video every record_every episodes


directory = os.path.realpath(__file__).split('/')
directory = '/'.join(directory[:-1])
replay_dir = directory + '/replays'
record_dir = directory + '/record'
assert_exists(replay_dir)
assert_exists(record_dir)

train_env = Manipulation_Env()

obs = train_env.reset()
done = False

length_history = []
ep_length = 0
reward_history = []
ep_reward = 0
ep_num = 0

# TODO: write resolution in common configuration
data_specs = (specs.Array((480,480,3), np.float32, 'observation'),
              specs.Array((5,), np.float32, 'action'),
              specs.Array((1,), np.float32, 'reward'),
              specs.Array((1,), np.float32, 'discount'))
replay_storage = ReplayBufferStorage(data_specs, replay_dir)
replay_loader = make_replay_loader(
    replay_dir, 1000000, 256, 4, False,
     3, 0.99
)

agent = DrQV2Agent((480,480,3), (5,), 'cuda', 1e-4, 50, 1024, 0.01, 2000,
    2, 'linear(1.0,0.1,500000)', 0.3, True)

for i in range(num_train_frames):
    if(done):  # An episode is done
        reward_history.append(ep_reward)
        length_history.append(ep_length)
        ep_length = 0
        ep_reward = 0
        obs = train_env.reset()

        if((ep_num % record_every == 0) and (ep_num > 0)):
            record.release()

        step = TimeStep()
        step.step_type = StepType.FIRST
        step.reward = 0
        step.discount = None
        step.observation = obs
        ep_num += 1

        replay_storage.add(step)

        if(ep_num % record_every == 0):
            record = cv2.VideoWriter(record_dir + '/' + str(ep_num) + '.mp4',
                cv2.VideoWriter_fourcc(*'mp4v', 30, (480,480)))
            record.write((obs * 255).astype(int))

    with torch.no_grad(), eval_mode(agent):
        action = agent.act(obs, i, eval_mode=False)
    if(i > 4000):
        metrics = agent.update(iter(replay_loader), i)
    obs, reward, done, info = train_env.step()
    ep_reward += reward

    step = TimeStep()
    step.reward = reward
    step.discount = 1
    step.observation = obs
    if(not done):
        step.step_type = StepType.MID
    else:
        step.step_type = StepType.LAST
    replay_storage.add(step)
    ep_length += 1
    if(ep_num % record_every == 0):
        record.write((obs * 255).astype(int))