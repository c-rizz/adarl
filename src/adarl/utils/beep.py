#!/usr/bin/env python3
from playsound import playsound
import os
import adarl.utils.dbg.ggLog as ggLog
from adarl.utils.utils import exc_to_str, pkgutil_get_path

initialized = False
pub = None

def init():
    # ggLog.info(f"beep init")
    try:
        import rospy
        from std_msgs.msg import String
        global pub
        pub = rospy.Publisher('/adarl_ros/beep', String, queue_size=10)
        pub.publish("")
    except:
        ggLog.warn(f"Failed to initialize beeper publisher")
        pub = None
    global initialized
    initialized = True

def beep(send_msg = True):
    # ggLog.info("beep")
    if not initialized:
        init()
    try:
        if send_msg:
            if pub is not None:
                pub.publish("beep")
            else:
                ggLog.warn(f"Tried to publish beep but publisher is disabled. This only works when using ROS.")
        playsound(pkgutil_get_path("adarl","assets/audio/beep.ogg"))
    except Exception as e:
        ggLog.info(f"Failed to beep: {exc_to_str(e)}")


def boop(send_msg = True):
    # ggLog.info("boop")
    if not initialized:
        init()
    try:
        if send_msg:
            if pub is not None:
                pub.publish("boop")
            else:
                ggLog.warn(f"Tried to publish boop but publisher is disabled. This only works when using ROS.")
        playsound(pkgutil_get_path("adarl","assets/audio/boop.ogg"))
    except Exception as e:
        ggLog.info(f"Failed to boop: {exc_to_str(e)}")
