import threading
import torch as th
from collections import deque
import queue
import atexit
from typing import Callable, Optional
import os
import adarl.utils.dbg.ggLog as ggLog
import time
import atexit



class BlockingPeekQueue:
    def __init__(self):
        self.queue = deque()
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)

    def put(self, item):
        # ggLog.info(f"put...")
        with self.not_empty:
            self.queue.append(item)
            self.not_empty.notify_all()
        # ggLog.info(f"put!")

    def get(self, timeout : float | None = None):
        # ggLog.info(f"get...")
        with self.not_empty:
            while len(self.queue) == 0:
                timedout = not self.not_empty.wait(timeout=timeout)
                if timedout:
                    raise queue.Empty()
            r = self.queue.popleft()
        # ggLog.info(f"get!")
        return r

    def peek(self, timeout : float | None = None):
        # ggLog.info(f"peek...")
        with self.not_empty:
            while len(self.queue) == 0:
                timedout = not self.not_empty.wait(timeout=timeout)
                if timedout:
                    # ggLog.info(f"peek! (empty)")
                    raise queue.Empty()
            r = self.queue[0]
        # ggLog.info(f"peek!")
        return r
        
    def __len__(self):
        return len(self.queue)
        
class Async_cuda2cpu_queue():
    def __init__(self):

        self._running = True
        self._worker_thread : Optional[threading.Thread] = None
        self._queue : BlockingPeekQueue | None = None
        atexit.register(self.close)

    def start_worker(self):
        """Starts a worker thread that can receive logs from a queue and submit them to wandb.
            Logs can be sent from other processes by sending to these processes the WandbWrapper
            object itself. When you call wandb_log from the child process it will recognize
            he is a child process and send the logs to the queue.
        """
        # ggLog.info(f"Starting Async_cuda2cpu_queue worker")
        # traceback.print_stack()
        self._queue = BlockingPeekQueue()
        self._worker_thread = threading.Thread(target=self._worker, name="WandbWrapper_worker")
        self._worker_thread.start()


    def _worker(self):
        import adarl.utils.session as session
        ggLog.info(f"Starting WandbWrapper worker in process {os.getpid()}")
        while not session.default_session.is_shutting_down() or not self._running:
            try:
                event, cuda_tensors, cpu_tensors, callback = self._queue.peek(timeout=1.0)
                if event.query():
                    self._queue.get(timeout=0)
                    callback(cpu_tensors)
                else:
                    time.sleep(0.01) # I believe using event.wait would block the entire python process (actually it would be nice if the wholw wandbwrapper was in a separate process from the rest)
            except queue.Empty as e:
                pass
        
    def send(self, cuda_tensors : dict[str,th.Tensor], callback : Callable[[dict[str,th.Tensor]], None]):
        cpu_tensors = {k:t.to(device="cpu", non_blocking=True) for k,t in cuda_tensors.items()}
        event = th.Event()
        event.record()
        self._queue.put((event, cuda_tensors, cpu_tensors, callback))

    def close(self):
        self._running = False
        if self._worker_thread is not None:
            self._worker_thread.join()
        if self._queue is not None and len(self._queue)>0:
            ggLog.warn(f"Async_tensor_cuda2cpu_queue closing with {len(self._queue)} tensors in queue")


    def __getstate__(self):
        state = self.__dict__.copy()  # Copy the object's state
        del state['_queue']  # Remove the attribute we don't want to pickle
        del state['_worker_thread']  # Remove the attribute we don't want to pickle
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._queue = None
        self._worker_thread = None

__singleton_queue_pid = None
__singleton_queue = None
def get_async_cuda2cpu_queue():
    global __singleton_queue
    global __singleton_queue_pid
    if __singleton_queue is None or __singleton_queue_pid != os.getpid():
        __singleton_queue = Async_cuda2cpu_queue()
        __singleton_queue.start_worker()
        atexit.register(lambda: __singleton_queue.close())
        __singleton_queue_pid = os.getpid()
    return __singleton_queue
    
def run_async_job(tensors : dict[str,th.Tensor], callback : Callable[[dict[str,th.Tensor]], None]):
    get_async_cuda2cpu_queue().send(tensors, callback)


def log_async(string : str, tensors : dict[str,th.Tensor], loglevel = "info"):
    def callback(tensors : dict[str,th.Tensor]):
        nonlocal string
        string = string.format(tensors)
        # for k,t in tensors:
        #     string = string.replace("{"+k+"}", f"{t}")
        getattr(ggLog,loglevel)(string)
    get_async_cuda2cpu_queue().send(tensors, callback)