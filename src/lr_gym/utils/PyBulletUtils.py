
import pybullet as p
import pybullet_data
import lr_gym.utils.dbg.ggLog as ggLog
from pathlib import Path
import lr_gym.utils.utils
import time
import pkgutil
egl = pkgutil.get_loader('eglRenderer')

client_id = None

def start():
    """
    Start Pyullet simulation.

    This ends up calling examples/SharedMemory/PhysicsServerCommandProcessor.cpp:createEmptyDynamicsWorld()
    This means it uses a MultiBodyDynamicsWorld
    """
    global client_id
    client_id = p.connect(p.DIRECT)
    plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

def buildPlaneWorld():
    # Taken from pybullet's scene_abstract.py
    p.setGravity(0, 0, -9.8)
    p.setDefaultContactERP(0.9)
    #print("self.numSolverIterations=",self.numSolverIterations)
    p.setPhysicsEngineParameter(fixedTimeStep=0.0165 / 4 * 4,
                                numSolverIterations=5,
                                numSubSteps=4)

    ggLog.info("Physics engine parameters:"+str(p.getPhysicsEngineParameters()))

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeObjId = p.loadURDF("plane.urdf")


    # planeObjId = p.loadURDF(lr_gym.utils.utils.pkgutil_get_path("lr_gym","models/cube.urdf"), basePosition = [0,     0,0])
    
    # planeObjId = p.loadURDF(lr_gym.utils.utils.pkgutil_get_path("lr_gym","models/cube.urdf"), basePosition = [10,     0,0])
    
    # planeObjId = p.loadURDF(lr_gym.utils.utils.pkgutil_get_path("lr_gym","models/cube.urdf"), basePosition = [-10,    1,0])
    # planeObjId = p.loadURDF(lr_gym.utils.utils.pkgutil_get_path("lr_gym","models/cube.urdf"), basePosition = [-10,   -1,0])

    # planeObjId = p.loadURDF(lr_gym.utils.utils.pkgutil_get_path("lr_gym","models/cube.urdf"), basePosition = [-1.5,   10,0])
    # planeObjId = p.loadURDF(lr_gym.utils.utils.pkgutil_get_path("lr_gym","models/cube.urdf"), basePosition = [ 0,     10,0])
    # planeObjId = p.loadURDF(lr_gym.utils.utils.pkgutil_get_path("lr_gym","models/cube.urdf"), basePosition = [ 1.5,   10,0])
    
    # planeObjId = p.loadURDF(lr_gym.utils.utils.pkgutil_get_path("lr_gym","models/cube.urdf"), basePosition = [-2,    -10,0])
    # planeObjId = p.loadURDF(lr_gym.utils.utils.pkgutil_get_path("lr_gym","models/cube.urdf"), basePosition = [-0.75, -10,0])
    # planeObjId = p.loadURDF(lr_gym.utils.utils.pkgutil_get_path("lr_gym","models/cube.urdf"), basePosition = [ 0.75, -10,0])
    # planeObjId = p.loadURDF(lr_gym.utils.utils.pkgutil_get_path("lr_gym","models/cube.urdf"), basePosition = [ 2,    -10,0])

    # Taken from pybullet's scene_stadium.py
    p.changeDynamics(planeObjId, -1, lateralFriction=0.8, restitution=0.5)

    #Taken from env_bases.py (works both with and without)
    # p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

    return planeObjId



def unloadModel(object_id : int):
    p.removeBody(object_id)

def startupPlaneWorld():
    start()
    # ggLog.info("Started pybullet")
    buildPlaneWorld()

def destroySimpleEnv():
    p.resetSimulation()
    p.disconnect(client_id)


