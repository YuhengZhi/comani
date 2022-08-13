#python
def sysCall_init():
    if simROS:
        global sensor_itself
        global seq
        global publisher
        seq = 0
        sensor_itself = sim.getObject('.')
        publisher = simROS.advertise('/camera_image', 'sensor_msgs/Image')
        cur_image, cur_res = sim.getVisionSensorImg(sensor_itself)
        print(type(cur_image))
        print(cur_res)
        print(seq)
        
def sysCall_sensing():
    if simROS:
        global seq
        cur_image, cur_res = sim.getVisionSensorImg(sensor_itself)
        image_msg = {}
        header = {}
        header['seq'] = seq
        seq += 1
        header['stamp'] = sim.getSimulationTime()
        header['frame_id'] = 'default_camera'
        image_msg['header'] = header
        image_msg['width'] = cur_res[0]
        image_msg['height'] = cur_res[1]
        image_msg['encoding'] = 'rgb8'
        image_msg['is_bigendian'] = 1
        image_msg['step'] = cur_res[0] * 3
        image_msg['data'] = [int(cur_image[pixel]) for pixel in range(len(cur_image))]
        simROS.publish(publisher, image_msg)

def sysCall_cleanup():
    if simROS:
        simROS.shutdownPublisher(publisher)