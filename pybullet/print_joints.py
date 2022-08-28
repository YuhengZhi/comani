import pybullet as p
import pybullet_data

physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0.0, 0.0, 0.5]
startOrientation = p.getQuaternionFromEuler([0,0,0])
pandaId = p.loadURDF("franka_panda/panda.urdf", startPos, startOrientation)

for i in range(p.getNumJoints(pandaId)):
    print(p.getJointInfo(pandaId, i)[9])
p.disconnect()