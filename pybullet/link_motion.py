import pybullet as p
import time
import pybullet_data
import math

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0.0, 0.0, 0.03]
startOrientation = p.getQuaternionFromEuler([0,0,0])
seven_link = p.loadURDF("../urdf/seven_link.urdf", startPos, startOrientation, useFixedBase = 1)

def wait_some_seconds(seconds):
    for i in range(math.ceil(seconds * 24 * 10)):
        p.stepSimulation()
        time.sleep(1.0/240)

joint_numbers = [0, 1, 3, 5, 7, 9, 11]
p.setJointMotorControlArray(seven_link, joint_numbers, p.VELOCITY_CONTROL, targetVelocities = [0,0,0,0,0,0,0], forces=[200,200,20,20,20,20,20])

speed = 1
p.setJointMotorControl2(seven_link, joint_numbers[2], p.VELOCITY_CONTROL, targetVelocity=speed, force=20)
wait_some_seconds(0.5)
for i in range(30):
    speed = -speed
    p.setJointMotorControl2(seven_link, joint_numbers[2], p.VELOCITY_CONTROL, targetVelocity=speed, force=20)
    wait_some_seconds(0.5)
p.setJointMotorControl2(seven_link, joint_numbers[2], p.VELOCITY_CONTROL, targetVelocity=0, force=20)

time.sleep(5)

p.setJointMotorControl2(seven_link, joint_numbers[0], p.VELOCITY_CONTROL, targetVelocity=1, force=20)
wait_some_seconds(1)
for i in range(1, len(joint_numbers)):
    p.setJointMotorControlArray(seven_link, [joint_numbers[i-1], joint_numbers[i]], p.VELOCITY_CONTROL, targetVelocities = [0,1], forces=[20,20])
    wait_some_seconds(1)
p.setJointMotorControl2(seven_link, joint_numbers[-1], p.VELOCITY_CONTROL, targetVelocity=0, force=20)
wait_some_seconds(1)
p.disconnect()