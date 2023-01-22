import torch
import gym
import numpy as np
import cv2
import pickle
import faulthandler
import utils
from dm_env import specs
from dm_env import StepType
from replay_buffer import ReplayBufferStorage, make_replay_loader
from utils import eval_mode
from drqv2 import DrQV2Agent
from pathlib import Path
from dmc import ExtendedTimeStep
from logger import Logger
from torch.utils.tensorboard import SummaryWriter
from coppeliasim_gym_wrapper import CoppeliaSim_Env

faulthandler.enable()

# Initialize environment
env = CoppeliaSim_Env()

# Training configuration variables
num_train_frames    = 3100000 # Taken from the medium.yaml in cfgs -> task
eval_run            = False # If this run is an evaluation run
eval_episodes       = 5
record_every        = env.vid_record_int # Record a video as per the intervals set in the wrapper
save_every          = 100 # Save an agent snapshot every save_every episodes
update_metrics_int  = 5

load_from           = "" # Option to load a saved checkpoint

# Agent configuration variables
stddev_schedule     = 'linear(1.0,0.1,500000)' # Taken from the medium.yaml in cfgs -> task
learning_rate       = 1e-4
obs_shape           = (3,84,84)
action_shape        = env.action_space.shape
feature_dim         = 50
hidden_dim          = 1024
critic_target_tau   = 0.01
num_expl_steps      = 4000
num_seed_frames     = 4000
update_every_steps  = 2
stddev_clip         = 0.3
batch_size          = 256
use_tb              = True
metrics             = None
sw                  = None

# Create recording directories
directory           = Path.cwd()
replay_dir          = directory / "replays"
record_dir          = directory / "record"
save_dir            = directory / "snapshots"
metric_dir          = directory / "metrics"
record_dir.mkdir(exist_ok=True)
save_dir.mkdir(exist_ok=True)
metric_dir.mkdir(exist_ok=True)

# Setup logger
# logger = Logger(metric_dir, use_tb=True)
sw = SummaryWriter(log_dir = str(metric_dir))

# Function to log episode rewards in a text file
def log_line(line):
    print(line)
    log_file.write(line + '\n')

# Function to evaluate the snapshot captured
def evaluate():
    eval_dir    = directory / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    for i in range(eval_episodes):
        obs     = env.reset()
        record  = cv2.VideoWriter(str(eval_dir) + '/' + 'snapshot_2500_' + str(i) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (84,84))

        # Save Image and write it to VideoWriter
        img     = np.array(obs, dtype = np.uint8)
        # img = obs[:3,:,:]
        img.resize([84, 84, 3])
        record.write(img)

        done        = False
        ep_length   = 0
        ep_reward   = 0
        print("Evaluate episode " + str(i))

        while(not done):
            with torch.no_grad(), eval_mode(agent):
                action = agent.act(obs, i, eval_mode = True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_length += 1

            # Save Image and write it to VideoWriter
            img = np.array(obs, dtype = np.uint8)
            # img = obs[:3,:,:]
            img.resize([84, 84, 3])
            record.write(img)

        print("Reward " + str(ep_reward) + "  Length " + str(ep_length))
        record.release()

def log_metrics(metrics, steps):
    for (key, value) in metrics.items():
        sw.add_scalar(key, value, steps)

log_file    = open('training_log', 'w')

obs         = env.reset()
done        = False

# Local variables
length_history  = []
ep_length       = 0
reward_history  = []
ep_reward       = 0
ep_num          = 0

data_specs      = (specs.Array(obs_shape, np.uint8, 'observation'),
                    specs.Array(action_shape, np.float32, 'action'),
                    specs.Array((1,), np.float32, 'reward'),
                    specs.Array((1,), np.float32, 'discount'))
replay_storage  = ReplayBufferStorage(data_specs, replay_dir)
replay_loader   = make_replay_loader(
    replay_dir, 1000000, 256, 0, False,
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
    env.stop_simulation()
    quit()

log_line("Episode 0")

# Initiate video recording
record = cv2.VideoWriter(str(record_dir) + '/' + str(ep_num) + '.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'), 30, (84,84))

# Training loop
for i in range(num_train_frames):
    if(done):
        log_line("Reward " + str(ep_reward))
        reward_history.append(ep_reward)
        length_history.append(ep_length)
        ep_length   = 0
        ep_reward   = 0
        obs         = env.reset()

        if((ep_num % record_every == 0)):
            record.release()

        step = ExtendedTimeStep(
            step_type   = StepType.FIRST,
            reward      = 0,
            discount    = 1,
            observation = obs,
            action      = np.zeros(7, dtype=np.float32)
        )
        ep_num += 1

        replay_storage.add(step)

        log_line("Episode " + str(ep_num))
        if(ep_num % record_every == 0):
            record  = cv2.VideoWriter(str(record_dir) + '/' + str(ep_num) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (84,84))

            # Save Image and write it to VideoWriter
            img     = np.array(obs, dtype = np.uint8)
            # img = obs[:3,:,:]
            img.resize([84, 84, 3])
            record.write(img)

        # Save snapshot as per save_every step
        if(ep_num % save_every == 0):
            agent_file = save_dir / (str(ep_num)+'_save')
            with agent_file.open('wb') as f:
                torch.save(agent, agent_file)

    # Get new set of action from the agent
    with torch.no_grad(), eval_mode(agent):
        action = agent.act(obs, i, eval_mode=False)

    # try to update the agent
    # if (ep_num % update_metrics_int == 0) and (ep_num > 0) and done:
    if (i > num_seed_frames) and (i % update_every_steps == 0):
        metrics = agent.update(iter(replay_loader), i)

    # Print metrics as per update_metrics_int
    if (ep_num % update_metrics_int == 0) and (ep_num > 0) and done:
        log_metrics(metrics, i)
        print(metrics)

    obs, reward, done, info     = env.step(action)
    ep_reward                   += reward

    if(not done):
        step_status = StepType.MID
    else:
        step_status = StepType.LAST
    step = ExtendedTimeStep(
        step_type   = step_status,
        reward      = reward,
        discount    = 1,
        observation = obs,
        action      = action
    )
    replay_storage.add(step)
    ep_length += 1
    if((ep_num % record_every == 0)):
        # Save Image and write it to VideoWriter
        img = np.array(obs, dtype = np.uint8)
        # img = obs[:3,:,:]
        img.resize([84, 84, 3])
        record.write(img)

    local_metrics = {'episode_reward': ep_reward, 'episode_length': ep_length, 'episode': ep_num, 'buffer_size': len(replay_storage), 'step': i}
    log_metrics(local_metrics, i)

# Stop simulation and close log file once training is completed
env.stop_simulation()
log_file.close()
