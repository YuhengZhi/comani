import torch
import os
import numpy as np
import cv2
import pickle
import faulthandler
from dm_env import specs
from dm_env import StepType
from replay_buffer import ReplayBufferStorage, make_replay_loader
from pybullet_env import Manipulation_Env
from utils import eval_mode
from drqv2 import DrQV2Agent
from pathlib import Path
from dmc import ExtendedTimeStep
from matplotlib import pyplot as plt

faulthandler.enable()

# Configuration variables
num_train_frames = 1100000 # Taken from the medium difficulty rating
num_train_frames = 10000

eval_run = False # If this run is an evaluation run
eval_episodes = 10
record_every = 1 # Record a video every record_every episodes
save_every = 200 # Save an agent snapshot every save_every episodes

load_from = "" # Option to load a saved checkpoint

# Agent configuration variables
stddev_schedule = 'linear(1.0,0.1,100000)'
learning_rate = 1e-4
obs_shape = (3,84,84)
action_shape = (5,)
feature_dim = 50
hidden_dim = 1024
critic_target_tau = 0.01
num_expl_steps = 6000
update_every_steps = 2
stddev_clip = 0.3
use_tb = True


# Create recording directories
directory = Path.cwd()
replay_dir = directory / "replays"
record_dir = directory / "record"
save_dir = directory / "snapshots"
metric_dir = directory / "metrics"
record_dir.mkdir(exist_ok=True)
save_dir.mkdir(exist_ok=True)
metric_dir.mkdir(exist_ok=True)

def log_line(line):
    print(line)
    log_file.write(line + '\n')

def evaluate():
    eval_dir = directory / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    for i in range(eval_episodes):
        obs = train_env.reset()
        record = cv2.VideoWriter(str(eval_dir) + '/' + str(i) + '.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'), 30, (84,84))
        record.write((obs.transpose([1,2,0]) * 255).astype(np.uint8))
        done = False
        ep_length = 0
        ep_reward = 0
        print("Evaluate episode " + str(i))

        while(not done):
            with torch.no_grad(), eval_mode(agent):
                action = agent.act(obs, i, eval_mode = False)
            obs, reward, done, info = train_env.step(action)
            ep_reward += reward
            ep_length += 1
            record.write((obs.transpose([1,2,0]) * 255).astype(np.uint8))
        print("Reward " + str(ep_reward) + "  Length " + str(ep_length))
        record.release()

log_file = open('training_log', 'w')

# Initialize environment
train_env = Manipulation_Env()

obs = train_env.reset()
done = False

length_history = []
ep_length = 0
reward_history = []
ep_reward = 0
ep_num = 0

data_specs = (specs.Array(obs_shape, np.float32, 'observation'),
              specs.Array(action_shape, np.float32, 'action'),
              specs.Array((1,), np.float32, 'reward'),
              specs.Array((1,), np.float32, 'discount'))
replay_storage = ReplayBufferStorage(data_specs, replay_dir)
replay_loader = make_replay_loader(
    replay_dir, 100000, 256, 0, False,
     3, 0.99
)

agent = DrQV2Agent(obs_shape, action_shape, 'cuda', learning_rate,
    feature_dim, hidden_dim, critic_target_tau, num_expl_steps,
    update_every_steps, stddev_schedule, stddev_clip, use_tb)

if(not load_from == ""):
    log_line("Loading saved checkpoint from " + str(load_from))
    agent = torch.load(load_from)

if(eval_run):
    assert(not load_from == "")  # Assert that a save snapshot is specified
    evaluate()
    quit()

log_line("Episode 0")

for i in range(num_train_frames):
    if(done):  # An episode is done
        log_line("Reward " + str(ep_reward))
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

        log_line("Episode " + str(ep_num))
        if(ep_num % record_every == 0):
            record = cv2.VideoWriter(str(record_dir) + '/' + str(ep_num) + '.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'), 30, (84,84))
            record.write((obs.transpose([1,2,0]) * 255).astype(np.uint8))

        if(ep_num % save_every == 0):
            agent_file = save_dir / (str(ep_num)+'_save')
            with agent_file.open('wb') as f:
                torch.save(agent, agent_file)

    with torch.no_grad(), eval_mode(agent):
        action = agent.act(obs, i, eval_mode=False)
    if(i > num_expl_steps):
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

# Pickle reward history for possible analysis
with open(str(metric_dir) + '/' + 'reward_history_object', 'wb') as reward_file:
    pickle.dump(reward_history, reward_file, protocol = pickle.DEFAULT_PROTOCOL)

# Plot reward vs episode graph
fig, ax = plt.subplots(1,1)
ax.set_ylim([min(reward_history) - 2, 10])
ax.scatter([i+1 for i in range(len(reward_history))], reward_history, s=1)
plt.title("Total reward each episode")
plt.savefig(str(metric_dir) + '/' + 'reward_history.png')

log_file.close()
