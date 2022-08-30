import pybullet as p
import pybullet_data
import math
import cv2
import numpy as np

physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0.0, 0.0, 0.5]
startOrientation = p.getQuaternionFromEuler([0,0,0])
pandaId = p.loadURDF("franka_panda/panda.urdf", startPos, startOrientation)
baseConstraint = p.createConstraint(planeId, -1, pandaId, 0, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])

outVideo = cv2.VideoWriter('camera_2d.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920,1080))

pixelWidth = 1920
pixelHeight = 1080
aspect = 1920.0/1080
camDistance = 4
projection = p.computeProjectionMatrixFOV(60, aspect, 0.5, 5.0)

for i in range(1440):
    camPosition = [math.sin(math.pi*2*i/1440), math.cos(math.pi*2*i/1440), 4.0]
    targetPosition = camPosition[:2] + [0.0]
    view = p.computeViewMatrix(camPosition, targetPosition, [0.0, 1.0, 0.0])
    _, _, curimg, _, _ = p.getCameraImage(1920, 1080, view, projection)
    curimg = np.asarray(curimg)
    outVideo.write(np.stack([curimg[:, :, 2], curimg[:, :, 1], curimg[:, :, 0]], 2))
    print(i)

outVideo.release()
p.removeConstraint(baseConstraint)
p.disconnect()