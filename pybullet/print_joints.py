import pybullet as p
import pybullet_data

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0.0, 0.0, 0.5]
startOrientation = p.getQuaternionFromEuler([0,0,0])
two_link = p.loadURDF("../urdf/seven_link.urdf", startPos, startOrientation, useFixedBase = 1)

for i in range(p.getNumJoints(two_link)):
    print(p.getJointInfo(two_link, i))
p.disconnect()