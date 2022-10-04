import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.GUI)
initialPos = [0,0,0.03]
p.setGravity(0,0,-10)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
initialOrientation = p.getQuaternionFromEuler([0,0,0])
plane = p.loadURDF("plane.urdf")
two_link = p.loadURDF("../urdf/two_link.urdf", initialPos, initialOrientation)

target_sphere = p.loadURDF("../urdf/sphere_blue.urdf",
    [0,0.2,0.08], initialOrientation, globalScaling=0.05, useFixedBase=True)

for i in range(5):
    p.setCollisionFilterPair(two_link, target_sphere, i, 0, 0)

for i in range(480):
    p.setJointMotorControlArray(two_link, [0,1], p.VELOCITY_CONTROL, targetVelocities = [1,0], forces=[20,20])
    p.stepSimulation()
    time.sleep(1.0/24)
    print(p.getLinkState(two_link, 3)[0])

p.disconnect()