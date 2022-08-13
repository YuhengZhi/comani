#python
def subscriber_callback(msg):
    # This is the subscriber callback function
    sim.addLog(sim.verbosity_scriptinfos,'subscriber receiver following Float32: ' + str(msg['data']))

def sysCall_init():
    # The child script initialization
    objectHandle=sim.getObject('.')
    objectAlias=sim.getObjectAlias(objectHandle,3)
    global publisher
    global subscriber

    # Prepare the float32 publisher and subscriber (we subscribe to the topic we advertise):
    if simROS:
        publisher=simROS.advertise('/simulationTime','std_msgs/Float32')
        print(publisher)
        subscriber=simROS.subscribe('/simulationTime','std_msgs/Float32','subscriber_callback')

def sysCall_actuation():
    # Send an updated simulation time message, and send the transform of the object attached to this script:
    if simROS:
        print("activation")
        print(publisher)
        simROS.publish(publisher,{'data': sim.getSimulationTime()})
        # To send several transforms at once, use simROS.sendTransforms instead

def sysCall_cleanup():
    # Following not really needed in a simulation script (i.e. automatically shut down at simulation end):
    if simROS:
        simROS.shutdownPublisher(publisher)
        simROS.shutdownSubscriber(subscriber)
