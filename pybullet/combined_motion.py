import pybullet as p
import time
import pybullet_data
import math
import cv2
import numpy as np

def wait_observe():
    global frame
    for k in range(300):
        p.stepSimulation()
        time.sleep(1.0/240)
        camPosition = [math.sin(math.pi*2*frame/1440), math.cos(math.pi*2*frame/1440), 4.0]
        targetPosition = camPosition[:2] + [0.0]
        view = p.computeViewMatrix(camPosition, targetPosition, [0.0, 1.0, 0.0])
        _, _, curimg, _, _ = p.getCameraImage(1920, 1080, view, projection)
        curimg = np.asarray(curimg)
        outVideo.write(np.stack([curimg[:, :, 2], curimg[:, :, 1], curimg[:, :, 0]], 2))
        frame += 1

physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0.0, 0.0, 0.5]
startOrientation = p.getQuaternionFromEuler([0,0,0])
pandaId = p.loadURDF("franka_panda/panda.urdf", startPos, startOrientation)
baseConstraint = p.createConstraint(planeId, -1, pandaId, 0, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])

outVideo = cv2.VideoWriter('camera_2d_combined.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920,1080))

pixelWidth = 1920
pixelHeight = 1080
aspect = 1920.0/1080
camDistance = 4
projection = p.computeProjectionMatrixFOV(60, aspect, 0.5, 5.0)

frame = 0

p.setJointMotorControl2(pandaId, 0, p.VELOCITY_CONTROL,
    targetVelocity = 1, force = 20)
wait_observe()
for i in range(1,7):
    p.setJointMotorControlArray(pandaId, [i-1, i], p.VELOCITY_CONTROL,
    targetVelocities = [0,1], forces = [20,20])
    wait_observe()
p.setJointMotorControl2(pandaId, 6, p.VELOCITY_CONTROL,
    targetVelocity = 0, force = 20)
wait_observe()

outVideo.release()
p.removeConstraint(baseConstraint)
p.disconnect()