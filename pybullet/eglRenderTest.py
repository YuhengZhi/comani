import pybullet as p
import time
import pkgutil
egl = pkgutil.get_loader('eglRenderer')
import pybullet_data
import cv2
import numpy as np

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
print("plugin=", plugin)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf", [0, 0, -1])
p.loadURDF("r2d2.urdf")

outVideo = cv2.VideoWriter("eglTest.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 220))

pixelWidth = 320
pixelHeight = 220
camTargetPos = [0, 0, 0]
camDistance = 4
pitch = -10.0
roll = 0
upAxisIndex = 2

for i in range(60):
  for yaw in range(0, 360, 10):
    start = time.time()
    p.stepSimulation()
    stop = time.time()
    print("stepSimulation %f" % (stop - start))

    #viewMatrix = [1.0, 0.0, -0.0, 0.0, -0.0, 0.1736481785774231, -0.9848078489303589, 0.0, 0.0, 0.9848078489303589, 0.1736481785774231, 0.0, -0.0, -5.960464477539063e-08, -4.0, 1.0]
    viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll,
                                                     upAxisIndex)
    projectionMatrix = [
        1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0, 0.0, 0.0, 0.0,
        -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0
    ]

    start = time.time()
    img_arr = p.getCameraImage(pixelWidth,
                               pixelHeight,
                               viewMatrix=viewMatrix,
                               projectionMatrix=projectionMatrix,
                               shadow=1,
                               lightDirection=[1, 1, 1])
    stop = time.time()
    print("renderImage %f" % (stop - start))
    #time.sleep(.1)
    #print("img_arr=",img_arr)
    img_arr = np.asarray(img_arr[2])
    outVideo.write(np.stack([img_arr[:,:,2], img_arr[:,:,1], img_arr[:,:,0]], axis=2))

outVideo.release()
p.unloadPlugin(plugin)
