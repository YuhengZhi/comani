import pybullet as p
import time
import pybullet_data

def wait_observe():
    for k in range(500):
        p.stepSimulation()
        time.sleep(1.0/240)

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,2]
startOrientation = p.getQuaternionFromEuler([0,0,0])
pandaId = p.loadURDF("franka_panda/panda.urdf", startPos, startOrientation)
baseConstraint = p.createConstraint(planeId, -1, pandaId, 0, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])

p.setJointMotorControl2(pandaId, 0, p.VELOCITY_CONTROL,
    targetVelocity = 1, force = 20)
wait_observe()
time.sleep(1)
for i in range(1,7):
    p.setJointMotorControlArray(pandaId, [i-1, i], p.VELOCITY_CONTROL,
    targetVelocities = [0,1], forces = [20,20])
    wait_observe()
    time.sleep(1)
p.setJointMotorControl2(pandaId, 6, p.VELOCITY_CONTROL,
    targetVelocity = 0, force = 20)
wait_observe()
time.sleep(10)
p.disconnect()