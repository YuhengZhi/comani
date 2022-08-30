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
#planeId = p.loadURDF("plane.urdf")
two_link = p.loadMJCF("mjcf/reacher.xml")

p.setJointMotorControl2(two_link[6], 0, p.VELOCITY_CONTROL,
    targetVelocity = 1, force = 20)
wait_observe()
time.sleep(1)
p.setJointMotorControlArray(two_link[6], [0, 2], p.VELOCITY_CONTROL,
targetVelocities = [0,1], forces = [20,20])
wait_observe()
time.sleep(1)
p.setJointMotorControl2(two_link[6], 2, p.VELOCITY_CONTROL,
    targetVelocity = 0, force = 20)
wait_observe()
time.sleep(10)
p.disconnect()