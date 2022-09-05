import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)

startPos = [0,0.2,1.0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
planeId = p.loadURDF("plane.urdf")
two_link = p.loadURDF("../reacher.urdf/reacher_link6_1.urdf", startPos, startOrientation)
for i in range(p.getNumJoints(two_link)):
    print(p.getJointInfo(two_link, i))
sphereId = p.loadURDF("sphere2red.urdf", startPos, startOrientation, globalScaling = 5)
for i in range(100):
    p.stepSimulation()
    time.sleep(1.0/240)
time.sleep(10)
p.disconnect()