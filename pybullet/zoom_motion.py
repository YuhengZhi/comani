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
plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
print("plugin=", plugin)

initialPos = [0,0,0.2]
initialOrientation = p.getQuaternionFromEuler([0,0,0])
sphere = p.loadURDF("sphere2red.urdf", initialPos, initialOrientation, globalScaling=0.2)
plane = p.loadURDF("plane.urdf")
initialPos = [0.1,-0.1,0.2]
shere_other = p.loadURDF("sphere_1cm.urdf", initialPos, initialOrientation)

view = p.computeViewMatrix([0,0,3],[0,0,0],[0,1,0])

outVideo = cv2.VideoWriter('sphere_compare.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920,1080))

pixelWidth = 1920
pixelHeight = 1080
aspect = 1920.0/1080
camDistance = 4
initialfov = 60
eventualfov = 40

for i in range(300):
    curfov = initialfov + (eventualfov - initialfov) * i / 300.0
    print(curfov)
    projection = p.computeProjectionMatrixFOV(curfov, aspect, 0.5, 5.0)

    _, _, curimg, _, _ = p.getCameraImage(1920, 1080, view, projection)
    curimg = np.asarray(curimg)
    outVideo.write(np.stack([curimg[:, :, 2], curimg[:, :, 1], curimg[:, :, 0]], 2))
    p.stepSimulation()
    time.sleep(1.0/240)

outVideo.release()
p.unloadPlugin(plugin)
p.disconnect()