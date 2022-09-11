import torch
import os
import numpy as np
import cv2
from dm_env import specs
from dm_env import StepType
from replay_buffer import ReplayBufferStorage, make_replay_loader
from pybullet_env import Manipulation_Env
from utils import eval_mode
from drqv2 import DrQV2Agent
from pathlib import Path
from dmc import ExtendedTimeStep

def assert_exists(directory):
    if(not os.path.exists(directory)):
        os.mkdir(directory)

# Configuration variables
num_train_frames = 3100000 # Taken from the medium difficulty rating
num_train_frames = 10000
action_repeat = 2 # Not 100% sure what this means yet. Probably related to multi-step return

record_every = 2 # Record a video every record_every episodes


directory = Path.cwd()
replay_dir = directory / "replays"
record_dir = directory / "record"
record_dir.mkdir(exist_ok=True)

train_env = Manipulation_Env()

obs = train_env.reset()
done = False

length_history = []
ep_length = 0
reward_history = []
ep_reward = 0
ep_num = 0

# TODO: write resolution in common configuration
data_specs = (specs.Array((3,84,84), np.float32, 'observation'),
              specs.Array((5,), np.float32, 'action'),
              specs.Array((1,), np.float32, 'reward'),
              specs.Array((1,), np.float32, 'discount'))
replay_storage = ReplayBufferStorage(data_specs, replay_dir)
replay_loader = make_replay_loader(
    replay_dir, 1000000, 256, 0, False,
     3, 0.99
)

agent = DrQV2Agent((3,84,84), (5,), 'cuda', 1e-4, 50, 1024, 0.01, 2000,
    2, 'linear(1.0,0.1,500000)', 0.3, True)

print("Episode 0")

for i in range(num_train_frames):
    if(done):  # An episode is done
        reward_history.append(ep_reward)
        length_history.append(ep_length)
        ep_length = 0
        ep_reward = 0
        obs = train_env.reset()

        if((ep_num % record_every == 0) and (ep_num > 0)):
            record.release()

        step = ExtendedTimeStep(
            step_type = StepType.FIRST,
            reward = 0,
            discount = 1,
            observation = obs,
            action = np.zeros(5, dtype=np.float32)
        )
        ep_num += 1

        replay_storage.add(step)

        print("Episode " + str(ep_num))
        if(ep_num % record_every == 0):
            record = cv2.VideoWriter(str(record_dir) + '/' + str(ep_num) + '.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'), 30, (84,84))
            record.write((obs.transpose([1,2,0]) * 255).astype(np.uint8))

    with torch.no_grad(), eval_mode(agent):
        action = agent.act(obs, i, eval_mode=False)
    if(i > 4100):
        metrics = agent.update(iter(replay_loader), i)
    obs, reward, done, info = train_env.step(action)
    ep_reward += reward

    if(not done):
        step_status = StepType.MID
    else:
        step_status = StepType.LAST
    step = ExtendedTimeStep(
        step_type = step_status,
        reward = reward,
        discount = 1,
        observation = obs,
        action = action
    )
    replay_storage.add(step)
    ep_length += 1
    if((ep_num % record_every == 0) and (ep_num > 0)):
        record.write((obs.transpose([1,2,0]) * 255).astype(np.uint8))