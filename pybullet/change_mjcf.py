import pybullet as p
import pybullet_data
import cv2
import numpy as np
import pkgutil
import time

physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
egl = pkgutil.get_loader('eglRenderer')
#plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
#print("plugin=", plugin)

reacher = p.loadMJCF("mjcf/reacher.xml")

pixelWidth = 640
pixelHeight = 480
aspect = 4.0/3
camDistance = 3

view = p.computeViewMatrix([0,0,camDistance], [0,0,0], [0,1,0])
projection = p.computeProjectionMatrixFOV(30, aspect, 0.5, 5.0)

outVideo = cv2.VideoWriter('reacher_reset.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (pixelWidth, pixelHeight))

target = reacher[7]

for i in range(100):
    _, _, curimg, _, _ = p.getCameraImage(pixelWidth, pixelHeight, view, projection)
    curimg = np.asarray(curimg)
    outVideo.write(np.stack([curimg[:, :, 2], curimg[:, :, 1], curimg[:, :, 0]], 2))
    p.stepSimulation()
    time.sleep(1.0/240)

targetPos = [0.1,0.1,0.1]
targetOrientation = p.getQuaternionFromEuler([0,0,0])
p.resetBasePositionAndOrientation(target, targetPos, targetOrientation)

p.changeVisualShape(target, 2, rgbaColor=[0.8,0.1,0.1,1.0])

for i in range(100):
    _, _, curimg, _, _ = p.getCameraImage(pixelWidth, pixelHeight, view, projection)
    curimg = np.asarray(curimg)
    outVideo.write(np.stack([curimg[:, :, 2], curimg[:, :, 1], curimg[:, :, 0]], 2))
    p.stepSimulation()
    time.sleep(1.0/240)
outVideo.release()
#p.unloadPlugin(plugin)
p.disconnect()