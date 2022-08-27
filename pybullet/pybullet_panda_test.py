import pybullet as p
import time
import pybullet_data

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
pandaId = p.loadURDF("franka_panda/panda.urdf", startPos, startOrientation)
for i in range(1000):
    p.stepSimulation()
    time.sleep(1.0/240)
time.sleep(20)
p.disconnect()