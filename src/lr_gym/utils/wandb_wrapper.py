import wandb
import lr_gym.utils.dbg.ggLog as ggLog
import time

req_count = 0
max_reqs_per_min = 50
last_send_to_server_times = [0]*max_reqs_per_min

last_sent_times_by_key = {}
sent_count = {}

def wandb_log(dict, throttle_period = 0):
    global req_count
    global last_send_to_server_times
    global max_reqs_per_min
    global last_sent_times_by_key
    global sent_count

    t = time.monotonic()
    keys = tuple(dict.keys())
    last_logged = last_sent_times_by_key.get(keys,0)

    # ggLog.info(f"[{str(keys)[:10]}]: t-last_logged = {t-last_logged}")
    if t-last_logged<throttle_period:
        # ggLog.info(f"[{str(keys)[:10]}]: Throttling")
        return
    # ggLog.info(f"[{str(keys)[:10]}]: Sending")

    t_50reqs_ago = last_send_to_server_times[(req_count+1)%max_reqs_per_min]
    if t-t_50reqs_ago<60:
        ggLog.warn(f"Exceeding wandb rate limit, skipping wandb_log for keys {list(dict.keys())}. Sent logs counters = {sent_count}")
        return
    last_send_to_server_times[req_count%max_reqs_per_min] = t
    req_count += 1
    last_sent_times_by_key[keys] = t
    sent_count[keys] = sent_count.get(keys,0) + 1
    try:
        wandb.log(dict)
    except Exception as e:
        ggLog.warn(f"wandb log failed with error: {e}")