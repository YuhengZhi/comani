import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)

two_link = p.loadMJCF('mjcf/reacher.xml')
print(len(two_link))
for k in range(len(two_link)):
    print(k)
    for i in range(p.getNumJoints(two_link[k])):
        print(p.getJointInfo(two_link[k], i))
startPos = [0,0.2,0.2]
startOrientation = p.getQuaternionFromEuler([0,0,0])
sphereId = p.loadURDF("sphere_1cm.urdf", startPos, startOrientation, globalScaling = 5)
for i in range(100):
    p.stepSimulation()
    time.sleep(1.0/240)
time.sleep(20)
p.disconnect()