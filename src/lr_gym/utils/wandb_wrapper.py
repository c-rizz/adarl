import wandb
import lr_gym.utils.dbg.ggLog as ggLog
import time

req_count = 0
max_reqs_per_min = 50
last_send_to_server_times = [0]*max_reqs_per_min

last_log_times_by_key = {}

def wandb_log(dict):
    global req_count
    global last_send_to_server_times
    global max_reqs_per_min
    global last_log_times_by_key

    key_hash = hash(tuple(dict.keys()))
    last_logged = last_log_times_by_key.get(key_hash,0)

    t_50reqs_ago = last_send_to_server_times[(req_count-1)%max_reqs_per_min]
    t = time.monotonic()
    if t-t_50reqs_ago<60:
        ggLog.warn(f"Exceeding wandb rate limit, skipping wandb_log for keys {list(dict.keys())}")
        return
    last_send_to_server_times[req_count%max_reqs_per_min] = t
    req_count += 1
    try:
        wandb.log(dict)
    except Exception as e:
        ggLog.warn(f"wandb log failed with error: {e}")