import wandb
import lr_gym.utils.dbg.ggLog as ggLog
import time

req_count = 0
max_reqs_per_min = 50
last_send_to_server_times = [0]*max_reqs_per_min

last_sent_times_by_key = {}
sent_count = {}

last_warn_time = 0.0

def compute_means_stds(tensors_dict):
    dms = {}
    for k,v in tensors_dict.items():
        dms[k+"_mean"] = v.mean()
        dms[k+"_std"] = v.std()
    tensors_dict.update(dms)
    return tensors_dict

def wandb_log(dict, throttle_period = 0):
    global req_count
    global last_send_to_server_times
    global max_reqs_per_min
    global last_sent_times_by_key
    global sent_count
    global last_warn_time

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
        if t-last_warn_time > 60:
            ggLog.warn(f"Exceeding wandb rate limit, skipping wandb_log for keys {list(dict.keys())}. Sent logs counters = {sent_count}")
            last_warn_time = t
        return
    last_send_to_server_times[req_count%max_reqs_per_min] = t
    req_count += 1
    last_sent_times_by_key[keys] = t
    sent_count[keys] = sent_count.get(keys,0) + 1
    try:
        wandb.log(dict)
    except Exception as e:
        ggLog.warn(f"wandb log failed with error: {e}")

def wandb_log_hists(d, throttle_period = 0):
    keys = tuple(d.keys())
    last_logged = last_sent_times_by_key.get(keys,0)
    t = time.monotonic()
    if t-last_logged<throttle_period:
        # ggLog.info(f"[{str(keys)[:10]}]: Throttling")
        return
    dict_cpu = {k:t.detach().to("cpu", non_blocking=True) for k,t in d.items()}
    dict_hists = {k:wandb.Histogram(t.numpy()) for k,t in dict_cpu.items()}
    wandb_log(dict_hists, throttle_period=throttle_period)