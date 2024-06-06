from typing import Callable
import numpy as np


import adarl.utils.dbg.ggLog as ggLog
import traceback
import adarl.utils.utils
from adarl.utils.dbg.web_video_streamer import VideoStreamerPublisher


class DbgImg:
    _ros_publishers = {}
    _initialized = False
    def init(self):
        self._initialized = True
        self._ros_publishers = {}
        self._videostream_publisher : VideoStreamerPublisher = None
        self._web_dbg = True
        try:
            from cv_bridge import CvBridge
            self._cv_bridge = CvBridge()
            import rospy
            import sensor_msgs.msg
            self._has_ros = False
        except ModuleNotFoundError as e:
            ggLog.warn(f"ROS is not present , will not publish debug images. exception = {e}")
            self._has_ros = False
        if self._web_dbg:
            self._videostream_publisher = VideoStreamerPublisher()


    def _addDbgStream(self, streamName : str):
        import rospy
        import sensor_msgs.msg
        if not self._initialized:
            self.init()
        if self._has_ros:
            pub = rospy.Publisher(streamName,sensor_msgs.msg.Image, queue_size = 10)
            self._ros_publishers[streamName] = pub

    def _removeDbgStream(self, streamName : str):
        if not self._initialized:
            self.init()
        if self._has_ros:
            self._ros_publishers.pop(streamName, None)

    def num_subscribers(self, streamName : str) -> int:
        s = 0
        if self._has_ros:
            if streamName not in self._ros_publishers:
                self._addDbgStream(streamName)
            s += self._ros_publishers[streamName].get_num_connections()
        if self._web_dbg:
            s += self._videostream_publisher.num_subscribers()
        return s

    def publishDbgImg(self, streamName : str,
                            encoding : str = "rgb8",
                            force_publish : bool =False,
                            img_callback : Callable[[], np.ndarray] = None):
        try:
            if not self._initialized:
                self.init()
            if self.num_subscribers(streamName) <= 0 and not force_publish:
                return
            img = img_callback()
            if img is None:
                ggLog.info("dbg_img.publishDbgImg(): No image provided, will not publish")
            img = adarl.utils.utils.imgToCvIntRgb(img_chw_rgb=img)
            
            if self._has_ros:
                if self._ros_publishers[streamName].get_num_connections()>0 or force_publish:                    
                    rosMsg = self._cv_bridge.cv2_to_imgmsg(img, "passthrough")
                    rosMsg.encoding = "rgb8"
                    self._ros_publishers[streamName].publish(rosMsg)
            
            if self._web_dbg:
                self._videostream_publisher.pub(stream_name=streamName, npimage=img)
        except Exception as e:
            ggLog.error("Ignored exception "+str(e)+"\n"+traceback.format_exc())

helper = DbgImg()
