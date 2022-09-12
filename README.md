# comani
Hand-eye co-manipulation for improved grasping success rate.

## File structure
py\_test\_scripts contains python test scripts with associated coppeliaSim environments
These each test one individual aspect needed for the environment

`pybullet_env.py` is the current Gym environment with pybullet, which describes
a task that involves using a two-link arm to push a ball,
with camera positioning and zoom features.

To run the code, copy over `drqv2.py`, `replay_buffer.py`, `utils.py` and `dmc.py` from the official
drqv2 implementation

The **replays** directory is automatically created by `drqv2_train.py` if it does
not exist yet. This is where replays are stored. 