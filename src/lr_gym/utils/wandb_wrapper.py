import wandb
import lr_gym.utils.dbg.ggLog as ggLog
from lr_gym.utils.utils import exc_to_str
import time
import torch as th
import os
import threading
import queue
import lr_gym.utils.mp_helper as mp_helper
import atexit
import traceback
from typing import Optional

class WandbWrapper():
    def __init__(self):
        self.req_count = 0
        self.max_reqs_per_min = 50
        self.last_send_to_server_times = [0.0]*self.max_reqs_per_min

        self.last_sent_times_by_key = {}
        self.sent_count = {}

        self.last_warn_time = 0.0
        self._init_pid = os.getpid()
        self._running = True
        self._queue = mp_helper.get_context().Queue()
        self._wandb_initialized = False
        self._worker_thread : Optional[threading.Thread] = None
        atexit.register(self.close)

    def start_worker(self):
        """Starts a worker thread that can receive logs from a queue and submit them to wandb.
            Logs can be sent from other processes by sending to these processes the WandbWrapper
            object itself. When you call wandb_log from the child process it will recognize
            he is a child process and send the logs to the queue.
        """
        self._worker_thread = threading.Thread(target=self._worker)
        self._worker_thread.start()


    def _worker(self):
        import lr_gym.utils.session as session
        ggLog.info(f"Starting WandbWrapper worker in process {os.getpid()}")
        while not session.default_session.is_shutting_down() or not self._running:
            try:
                log_dict, throttle_period = self._queue.get(block=True, timeout=10)
                wandb_log(log_dict, throttle_period)
            except queue.Empty as e:
                pass

    def wandb_init(self, **kwargs):
        if os.getpid()!=self._init_pid:
            raise RuntimeError(f"tried to init in a process different from the main one")
        wandb.init(**kwargs)
        self._wandb_initialized = True
        # self._worker_thread = threading.Thread(target=self._worker)
        # self._worker_thread.start()
        

    def wandb_log(self, log_dict, throttle_period = 0):
        if not self._wandb_initialized:
            ggLog.error(f"Called wandb_log, but wandb is not initialized. Skipping log.")
            traceback.print_stack()
        try:
            if os.getpid()==self._init_pid:
                # ggLog.info(f"wandbWrapper logging directly (initpid = {self._init_pid})")
                if callable(log_dict):            
                    log_dict = log_dict()
                
                t = time.monotonic()
                keys = tuple(log_dict.keys())
                last_logged = self.last_sent_times_by_key.get(keys,0)

                # ggLog.info(f"[{str(keys)[:10]}]: t-last_logged = {t-last_logged}")
                if t-last_logged<throttle_period:
                    # ggLog.info(f"[{str(keys)[:10]}]: Throttling")
                    return
                # ggLog.info(f"[{str(keys)[:10]}]: Sending")

                t_50reqs_ago = self.last_send_to_server_times[(self.req_count+1)%self.max_reqs_per_min]
                if t-t_50reqs_ago<60:
                    if t-self.last_warn_time > 60:
                        ggLog.warn(f"Exceeding wandb rate limit, skipping wandb_log for keys {list(log_dict.keys())}. Sent logs counters = {self.sent_count}")
                        self.last_warn_time = t
                    return
                self.last_send_to_server_times[self.req_count%self.max_reqs_per_min] = t
                self.req_count += 1
                self.last_sent_times_by_key[keys] = t
                self.sent_count[keys] = self.sent_count.get(keys,0) + 1
                try:
                    wandb.log(log_dict)
                except Exception as e:
                    ggLog.warn(f"wandb log failed with error: {exc_to_str(e)}")
            else:
                ggLog.info(f"wandb_log called from non-main process")
                # raise NotImplementedError()
                self._queue.put((log_dict, throttle_period), block=False)
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

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_worker_thread"]
        return state

    def __setstate__(self, state):
        state["_worker_thread"] = None
        self.__dict__.update(state)

default_wrapper = WandbWrapper()

def wandb_init(**kwargs):
    return default_wrapper.wandb_init(**kwargs)


def wandb_log(log_dict, throttle_period = 0):
    default_wrapper.wandb_log(log_dict, throttle_period)

def wandb_log_hists(d, throttle_period):
    default_wrapper.wandb_log_hists( d, throttle_period)

def compute_means_stds(tensors_dict):
    dms = {}
    for k,v in tensors_dict.items():
        dms[k+"_mean"] = v.mean()
        dms[k+"_std"] = v.std()
    tensors_dict.update(dms)
    return tensors_dict