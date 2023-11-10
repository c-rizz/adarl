
# from mujoco.egl import egl_ext as EGL

import ctypes
from OpenGL.platform import ctypesloader  # pylint: disable=g-bad-import-order
try:
  # Nvidia driver seems to need libOpenGL.so (as opposed to libGL.so)
  # for multithreading to work properly. We load this in before everything else.
  ctypesloader.loadLibrary(ctypes.cdll, 'OpenGL', mode=ctypes.RTLD_GLOBAL)
except OSError:
  pass

# pylint: disable=g-import-not-at-top
from OpenGL import EGL
from OpenGL import error
import OpenGL
import lr_gym.utils.dbg.ggLog as ggLog

# From the EGL_EXT_device_enumeration extension.
PFNEGLQUERYDEVICESEXTPROC = ctypes.CFUNCTYPE(
    EGL.EGLBoolean,
    EGL.EGLint,
    ctypes.POINTER(EGL.EGLDeviceEXT),
    ctypes.POINTER(EGL.EGLint),
)
try:
  _eglQueryDevicesEXT = PFNEGLQUERYDEVICESEXTPROC(  # pylint: disable=invalid-name
      EGL.eglGetProcAddress('eglQueryDevicesEXT'))
except TypeError as e:
  raise ImportError('eglQueryDevicesEXT is not available.') from e


# From the EGL_EXT_platform_device extension.
EGL_PLATFORM_DEVICE_EXT = 0x313F
PFNEGLGETPLATFORMDISPLAYEXTPROC = ctypes.CFUNCTYPE(
    EGL.EGLDisplay, EGL.EGLenum, ctypes.c_void_p, ctypes.POINTER(EGL.EGLint))
try:
  eglGetPlatformDisplayEXT = PFNEGLGETPLATFORMDISPLAYEXTPROC(  # pylint: disable=invalid-name
      EGL.eglGetProcAddress('eglGetPlatformDisplayEXT'))
except TypeError as e:
  raise ImportError('eglGetPlatformDisplayEXT is not available.') from e


# Wrap raw _eglQueryDevicesEXT function into something more Pythonic.
def eglQueryDevicesEXT(max_devices=10):  # pylint: disable=invalid-name
  devices = (EGL.EGLDeviceEXT * max_devices)()
  num_devices = EGL.EGLint()
  success = _eglQueryDevicesEXT(max_devices, devices, num_devices)
  if success == EGL.EGL_TRUE:
    return [devices[i] for i in range(num_devices.value)]
  else:
    raise error.GLError(err=EGL.eglGetError(),
                        baseOperation=eglQueryDevicesEXT,
                        result=success)

def isDisplayInitialized(display):
    try:
        OpenGL.EGL.eglQueryString(display, OpenGL.EGL.EGL_VENDOR)
        initialized = True
    except OpenGL.error.GLError as e:
        # if e.err.name == "EGL_NOT_INITIALIZED":
        initialized = False
    return initialized

def getDisplayInfos():
    devices = eglQueryDevicesEXT()
    ggLog.info(f"Got {len(devices)} egl devices")
    infos = []
    count = 0
    for device in devices:
        display = eglGetPlatformDisplayEXT( EGL_PLATFORM_DEVICE_EXT,device,None)
        ggLog.info(f"display={display}")
        was_initialized = isDisplayInitialized(display)
        ggLog.info(f"was_initialized={was_initialized}")
        try:
            OpenGL.EGL.eglInitialize(display,None,None)
            vendor = OpenGL.EGL.eglQueryString(display, OpenGL.EGL.EGL_VENDOR)
            version = OpenGL.EGL.eglQueryString(display, OpenGL.EGL.EGL_VERSION)
            client_apis = OpenGL.EGL.eglQueryString(display, OpenGL.EGL.EGL_CLIENT_APIS)
            extensions = OpenGL.EGL.eglQueryString(display, OpenGL.EGL.EGL_EXTENSIONS)
            if not was_initialized:
                OpenGL.EGL.eglTerminate(display)
            infos.append({"id":count,
                        "vendor":vendor,
                        "egl_version":version,
                        "egl_client_apis":client_apis,
                        "extensions":extensions})
        except OpenGL.error.GLError:
            pass
        count += 1
    infosstr = '\n'.join([str(info)for info in infos])
    ggLog.info(f"Available egl devices: {infosstr}")
    return infos

def getEglDeviceIdsByVendor(vendor):
    return [info["id"] for info in getDisplayInfos() if info["vendor"].decode('UTF-8')==vendor]