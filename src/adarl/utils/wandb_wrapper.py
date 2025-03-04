import adarl.utils.dbg.ggLog as ggLog
from adarl.utils.utils import exc_to_str, get_caller_info
import time
import torch as th
import os
os.environ["WANDB_ERROR_REPORTING"] = "False" # Having this enabled leavs some threads up at the end
import wandb
import threading
import queue
import adarl.utils.mp_helper as mp_helper
import atexit
import traceback
from typing import Optional
from adarl.utils.tensor_trees import is_all_finite, non_finite_flat_keys, map_tensor_tree
import numpy as np
from typing import Callable
from collections import deque



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


def _fix_histogram_range(value):
    """ In case there are nan/inf values in an array/tensor wandb drops the entire log, because it fails to
        detect the range. This doesnt make much sense.
        So I compute the range myself, excuding infs and nans, and produce a numpy histogram.
    """
    if isinstance(value, (th.Tensor | np.ndarray)):
        if not isinstance(value, th.Tensor):
            value = th.as_tensor(value)
        if value.ndim>0 and len(value) > 1:
            # finite_values = value[th.isfinite(value)]
            value = th.histogram(value) #, range=(finite_values.min(), finite_values.max()))
        else:
            return value
    else:
        return value


class WandbWrapper():
    def __init__(self):
        self.req_count = 0
        self.max_reqs_per_min = 40
        self.last_send_to_server_times = [0.0]*self.max_reqs_per_min

        self.last_sent_times_by_key = {}
        self.sent_count = {}

        self.last_warn_time = 0.0
        self._running = True
        self._mp_queue = mp_helper.get_context().Queue()
        self._wandb_initialized = False
        self._worker_thread : Optional[threading.Thread] = None
        self._async_c2c_queue = Async_cuda2cpu_queue()
        atexit.register(self.close)

    def _start(self):
        """Starts a worker thread that can receive logs from a queue and submit them to wandb.
            Logs can be sent from other processes by sending to these processes the WandbWrapper
            object itself. When you call wandb_log from the child process it will recognize
            he is a child process and send the logs to the queue.
        """
        self._async_c2c_queue.start_worker()
        self._worker_thread = threading.Thread(target=self._worker, name="WandbWrapper_worker")
        self._worker_thread.start()


    def _worker(self):
        import adarl.utils.session as session
        ggLog.info(f"Starting WandbWrapper worker in process {os.getpid()}")
        while not session.default_session.is_shutting_down() or not self._running:
            try:
                log_dict, throttle_period, silent_throttling = self._mp_queue.get(block=True, timeout=1)
                wandb_log(log_dict, throttle_period, silent_throttling)
            except queue.Empty as e:
                pass

    def wandb_init(self, **kwargs):
        self._start()
        if self._wandb_initialized:
            raise RuntimeError(f"Tried to initialize wandb wrapper twice (original pid {self._init_pid}, current pid = {os.getpid()}")
        self._init_pid = os.getpid()
        wandb.require("core")
        wandb.init(**kwargs)
        self._wandb_initialized = True
        # self._worker_thread = threading.Thread(target=self._worker)
        # self._worker_thread.start()
    
    @staticmethod
    def _safe_wandb_log(log_dict : dict[str,th.Tensor]):
        log_dict = map_tensor_tree(log_dict, _fix_histogram_range)
        try:
            wandb.log(log_dict)
        except Exception as e:
            ggLog.warn(f"wandb log failed with error: {exc_to_str(e)}")
        
    def _async_thread_wandb_log(self, log_dict : dict[str, th.Tensor]):
        log_dict = map_tensor_tree(log_dict, lambda l: th.as_tensor(l))
        self._async_c2c_queue.send(log_dict, self._safe_wandb_log)


    def wandb_log(self, log_dict : dict[str, th.Tensor], throttle_period = 0, silent_throttling : bool = False):
        with th.no_grad():
            if not self._wandb_initialized:
                ggLog.warn(f"Called wandb_log, but wandb is not initialized. Skipping log.")
                # traceback.print_stack()
                return
            try:
                log_dict = map_tensor_tree(log_dict, lambda l: l.detach() if isinstance(l, th.Tensor) else l)
                # if not is_all_finite(log_dict):
                #     ggLog.warn(f"Non-finite values in wandb log. \n"
                #             f"Non-finite keys = {non_finite_flat_keys(log_dict)} \n"
                #             f"Stacktrace:\n{''.join(traceback.format_stack())}")

                if os.getpid()==self._init_pid:
                    # ggLog.info(f"wandbWrapper logging directly (initpid = {self._init_pid})")
                    if callable(log_dict):            
                        log_dict = log_dict()
                    
                    t = time.monotonic()
                    keys = tuple(log_dict.keys())
                    last_logged = self.last_sent_times_by_key.get(keys,0)

                    # ggLog.info(f"[{str(keys)[:10]}]: t-last_logged = {t-last_logged}")
                    if t-last_logged<throttle_period:
                        if not silent_throttling:
                            ggLog.info(f"wandb_log throttling ({t-last_logged}<{throttle_period}) {get_caller_info(depth=1, width=2, inline=True)}")
                        return
                    # ggLog.info(f"[{str(keys)[:10]}]: Sending")

                    t_50reqs_ago = self.last_send_to_server_times[(self.req_count+1)%self.max_reqs_per_min]
                    if t-t_50reqs_ago<60:
                        if t-self.last_warn_time > 60:
                            ggLog.warn(f"Exceeding wandb rate limit, skipping wandb_log from {get_caller_info(depth=1, width=2, inline=True)}. Sent logs counters = {self.sent_count}")
                            self.last_warn_time = t
                        return
                    self.last_send_to_server_times[self.req_count%self.max_reqs_per_min] = t
                    self.req_count += 1
                    self.last_sent_times_by_key[keys] = t
                    self.sent_count[keys] = self.sent_count.get(keys,0) + 1
                    import adarl.utils.session as session
                    log_dict["session_collected_steps"] = session.default_session.run_info["collected_steps"].value
                    log_dict["session_collected_episodes"] = session.default_session.run_info["collected_episodes"].value
                    log_dict["session_train_iterations"] = session.default_session.run_info["train_iterations"].value
                    self._async_thread_wandb_log(log_dict)
                    # log_dict = map_tensor_tree(log_dict, _fix_histogram_range)
                    # try:
                    #     wandb.log(log_dict)
                    # except Exception as e:
                    #     ggLog.warn(f"wandb log failed with error: {exc_to_str(e)}")
                else:
                    # ggLog.info(f"wandb_log called from non-main process")
                    # raise NotImplementedError()
                    self._mp_queue.put((log_dict, throttle_period, silent_throttling), block=False)
            except Exception as e:
                ggLog.warn(f"wandb_log failed with error: {exc_to_str(e)}")

    def wandb_log_hists(self, d, throttle_period = 0):
        keys = tuple(d.keys())
        last_logged = self.last_sent_times_by_key.get(keys,0)
        t = time.monotonic()
        if t-last_logged<throttle_period:
            # ggLog.info(f"[{str(keys)[:10]}]: Throttling")
            return
        dict_cpu = {k:t.detach().to("cpu", non_blocking=True) for k,t in d.items()}
        th.cuda.current_stream().synchronize() #Wait for non_blocking transfers (they are not automatically synchronized when used as inputs! https://discuss.pytorch.org/t/how-to-wait-on-non-blocking-copying-from-gpu-to-cpu/157010/2)
        dict_hists = {k:wandb.Histogram(t.numpy()) for k,t in dict_cpu.items()}
        self.wandb_log(dict_hists, throttle_period=throttle_period)

    def close(self):
        self._running = False
        if self._worker_thread is not None:
            self._worker_thread.join()
        self._async_c2c_queue.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_worker_thread"]
        return state

    def __setstate__(self, state):
        state["_worker_thread"] = None
        self.__dict__.update(state)

    def is_main_wandb_thread(self):
        return self._wandb_initialized and self._init_pid == os.getpid()

default_wrapper = WandbWrapper()

def wandb_init(**kwargs):
    return default_wrapper.wandb_init(**kwargs)


def wandb_log(log_dict, throttle_period = 0, silent_throttling : bool = False):
    # ggLog.info(f"wandb_log called at {traceback.format_list(traceback.extract_stack(limit=2))[0]}")
    default_wrapper.wandb_log(log_dict, throttle_period, silent_throttling)

def wandb_log_hists(d, throttle_period):
    default_wrapper.wandb_log_hists( d, throttle_period)

def compute_means_stds(tensors_dict):
    dms = {}
    for k,v in tensors_dict.items():
        dms[k+"_mean"] = v.mean()
        dms[k+"_std"] = v.std()
    tensors_dict.update(dms)
    return tensors_dict