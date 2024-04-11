import os
import numpy as np
import time
import lr_gym.utils.dbg.ggLog as ggLog
import io
import threading
import pathlib
import copy
import atexit
import stat
from typing import Dict
import zmq

class VideoStreamerPublisher():
    def __init__(self):
        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.PUB)
        self._zmq_socket.setsockopt(zmq.CONFLATE, 1)  # last msg only.
        self._zmq_socket.connect("ipc:///tmp/lr_gym/VideoStreamerPublisher")
        atexit.register(self._cleanup)


    def pub(self, stream_name, npimage : np.ndarray):
        with io.BytesIO() as blob:
            np.save(file=blob, arr=npimage) #could also use savez_compressed, but it's just a memeory copy I hope
            topic_name = stream_name.encode("utf-8")
            #we send aa s single message because wmq.CONFLATE does not work for multipart and we want to use it
            self._zmq_socket.send(  len(topic_name).to_bytes(length=4,byteorder="big")+
                                    topic_name+
                                    blob.getvalue())
            # ggLog.info(f"published")

    def _cleanup(self):
        self._zmq_socket.close()
        self._zmq_context.term()

    def num_subscribers(self):
        return 1 # TODO: placeholder


class VideoStreamerSubscriber():
    def __init__(self):
        self._alive = True
        self._rlock = threading.RLock()
        self._last_imgs = {}
        self._read_freq = 0.01

        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.SUB)
        self._zmq_socket.setsockopt(zmq.CONFLATE, 1)  # last msg only.
        self._zmq_socket.subscribe(b"") #subscribe to anything
        self._zmq_socket.bind("ipc:///tmp/lr_gym/VideoStreamerPublisher")

        self._listener_thread = threading.Thread(target=self._listener, name="VideoStreamerSubscriber_listener")
        self._listener_thread.start()
        atexit.register(self._cleanup)
        # ggLog.info(f"Started")


    def _listener(self):
        running = True
        while running:
            t0 = time.monotonic()
            # with self._rlock:
            #     running = self._alive
            rec = self._zmq_socket.recv()
            t = time.monotonic()
            topic_name_length = int.from_bytes(rec[:4],byteorder="big")
            topic = rec[4:4+topic_name_length].decode("utf-8")
            # ggLog.info(f"Got image for {topic}")
            with io.BytesIO(rec[4+topic_name_length:]) as b:
                nparr = np.load(b)
            self._last_imgs[topic] = (t, topic, nparr)
            # ggLog.info(f"Got image for {topic}: {nparr.shape}")

            rt = self._read_freq - (time.monotonic()-t0)
            if rt > 0:
                time.sleep(rt)

    def get_images(self):
        return copy.deepcopy(self._last_imgs)
    
    def _cleanup(self):
        with self._rlock:
            self._alive = False
        self._listener_thread.join()
        self._zmq_socket.close()
        self._zmq_context.term()

            



if __name__ == "__main__":
    pvs = VideoStreamerPublisher()
    pvs.add_stream("test")

    while True:
        a = np.random.randint(low = 0, high = 255, size=(200,200,3), dtype = np.uint8)
        pvs.pub("test",a)
        time.sleep(0.1)

