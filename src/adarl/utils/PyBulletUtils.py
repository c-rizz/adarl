
import pybullet as p
import pybullet_data
import adarl.utils.dbg.ggLog as ggLog
from pathlib import Path
import adarl.utils.utils
import time
import pkgutil
egl = pkgutil.get_loader('eglRenderer')
import threading

client_id = None
starter_thread = None

def start(debug_gui : bool = False):
    """
    Start Pyullet simulation.

    This ends up calling examples/SharedMemory/PhysicsServerCommandProcessor.cpp:createEmptyDynamicsWorld()
    This means it uses a MultiBodyDynamicsWorld
    """
    global client_id
    global starter_thread
    if debug_gui:
        client_id = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    else:
        client_id = p.connect(p.DIRECT)
        plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    starter_thread = threading.current_thread()

def buildPlaneWorld():
    # Taken from pybullet's scene_abstract.py
    p.setGravity(0, 0, -9.8)
    # p.setDefaultContactERP(0.9)
    #print("self.numSolverIterations=",self.numSolverIterations)
    p.setPhysicsEngineParameter( #fixedTimeStep=0.0165 / 4 * 4,
                                numSolverIterations=5,
                                numSubSteps=1, # using substeps breakks contacts detection (as the funciton only returns the last substep information)
                                enableFileCaching=0)

    ggLog.info("Physics engine parameters:"+str(p.getPhysicsEngineParameters()))

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeObjId = p.loadURDF("plane.urdf")


    # planeObjId = p.loadURDF(adarl.utils.utils.pkgutil_get_path("adarl","models/cube.urdf"), basePosition = [0,     0,0])
    
    # planeObjId = p.loadURDF(adarl.utils.utils.pkgutil_get_path("adarl","models/cube.urdf"), basePosition = [10,     0,0])
    
    # planeObjId = p.loadURDF(adarl.utils.utils.pkgutil_get_path("adarl","models/cube.urdf"), basePosition = [-10,    1,0])
    # planeObjId = p.loadURDF(adarl.utils.utils.pkgutil_get_path("adarl","models/cube.urdf"), basePosition = [-10,   -1,0])

    # planeObjId = p.loadURDF(adarl.utils.utils.pkgutil_get_path("adarl","models/cube.urdf"), basePosition = [-1.5,   10,0])
    # planeObjId = p.loadURDF(adarl.utils.utils.pkgutil_get_path("adarl","models/cube.urdf"), basePosition = [ 0,     10,0])
    # planeObjId = p.loadURDF(adarl.utils.utils.pkgutil_get_path("adarl","models/cube.urdf"), basePosition = [ 1.5,   10,0])
    
    # planeObjId = p.loadURDF(adarl.utils.utils.pkgutil_get_path("adarl","models/cube.urdf"), basePosition = [-2,    -10,0])
    # planeObjId = p.loadURDF(adarl.utils.utils.pkgutil_get_path("adarl","models/cube.urdf"), basePosition = [-0.75, -10,0])
    # planeObjId = p.loadURDF(adarl.utils.utils.pkgutil_get_path("adarl","models/cube.urdf"), basePosition = [ 0.75, -10,0])
    # planeObjId = p.loadURDF(adarl.utils.utils.pkgutil_get_path("adarl","models/cube.urdf"), basePosition = [ 2,    -10,0])

    # Taken from pybullet's scene_stadium.py
    p.changeDynamics(planeObjId, -1, lateralFriction=0.8, restitution=0.5)

    #Taken from env_bases.py (works both with and without)
    # p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

    return planeObjId



def unloadModel(object_id : int):
    p.removeBody(object_id)


def startupPlaneWorld(debug_gui : bool = False):
    start(debug_gui = debug_gui)
    # ggLog.info("Started pybullet")
    buildPlaneWorld()

def destroySimpleEnv():
    p.resetSimulation()
    p.disconnect(client_id)


