import OpenGL
from mujoco.egl import egl_ext as EGL
import lr_gym.utils.dbg.ggLog as ggLog

def isDisplayInitialized(display):
    try:
        OpenGL.EGL.eglQueryString(display, OpenGL.EGL.EGL_VENDOR)
        initialized = True
    except OpenGL.error.GLError as e:
        # if e.err.name == "EGL_NOT_INITIALIZED":
        initialized = False
    return initialized

def getDisplayInfos():
    devices = EGL.eglQueryDevicesEXT()
    infos = []
    count = 0
    for device in devices:
        display = EGL.eglGetPlatformDisplayEXT( EGL.EGL_PLATFORM_DEVICE_EXT,device,None)
        was_initialized = isDisplayInitialized(display)
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