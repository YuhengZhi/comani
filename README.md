# comani
Hand-eye co-manipulation for improved grasping success rate.

## File structure
py\_test\_scripts contains python test scripts with associated coppeliaSim environments
These each test one individual aspect needed for the environment

`pybullet_env.py` is the current Gym environment with pybullet, which describes
a task that involves using a two-link arm to push a ball,
with camera positioning and zoom features.

`drqv2.py` and `replay_buffer.py` are copied from the official drqv2 implementation
to provide the drqv2 agent and replay buffer

The **replays** directory is automatically created by `drqv2_train.py` if it does
not exist yet. This is where replays are stored. 