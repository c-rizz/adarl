from __future__ import annotations
import functools
import math

from adarl.utils.dbg.dbg_checks import dbg_check_size, dbg_check
import numpy as np
import time
from typing import List, Tuple, Callable, Dict, Union, Optional, Any, Optional, TypeVar, Sequence
import os
import quaternion
import tqdm

import adarl.utils.dbg.ggLog as ggLog
import torch as th
from dataclasses import dataclass
from adarl.utils.base_utils import *
import functools
import gymnasium as gym

numpy_to_torch_dtype_dict = {
    bool          : th.bool,
    np.uint8      : th.uint8,
    np.int8       : th.int8,
    np.int16      : th.int16,
    np.int32      : th.int32,
    np.int64      : th.int64,
    np.float16    : th.float16,
    np.float32    : th.float32,
    np.float64    : th.float64,
    np.complex64  : th.complex64,
    np.complex128 : th.complex128
}
numpy_to_torch_dtype_dict.update({np.dtype(npd):td for npd,td in numpy_to_torch_dtype_dict.items()})

torch_to_numpy_dtype_dict = {v:k for k,v in numpy_to_torch_dtype_dict.items()}

def thtens(array, device:th.device | str | None = None, dtype: th.dtype | None = None) -> th.Tensor:
    """Allocate a torch tensor from an array, just like torch.as_tensor, but avoids CUDA syncs

    Parameters
    ----------
    array : _type_
        _description_
    device : th.device | str | None, optional
        _description_, by default None
    dtype : th.dtype | None, optional
        _description_, by default None

    Returns
    -------
    th.Tensor
        _description_
    """
    if isinstance(array, th.Tensor):
        if dtype is not None:
            array = array.to(dtype=dtype)
        if device is not None:
            device = th.device(device) if isinstance(device, str) else device
            array = array.to(device=device, non_blocking=device.type=="cuda")
        return array
    if device is None:
        device = th.device("cpu")
    elif isinstance(device, str):
        device = th.device(device)
    return th.tensor(array, dtype=dtype).to(device=device,non_blocking=device.type=="cuda")


T = TypeVar('T')
@dataclass
class Pose:
    position : th.Tensor
    orientation_xyzw : th.Tensor

    def array_xyz_xyzw(self, type : type[T] = th.Tensor) -> T:
        tensor = th.concat([self.position,self.orientation_xyzw])
        if type == th.Tensor:
            return tensor
        elif type == np.ndarray:
            return tensor.cpu().numpy()
        elif type == list:
            return tensor.cpu().tolist()
        elif type == tuple:
            return tuple(tensor.cpu().to())

def build_pose(x,y,z, qx,qy,qz,qw, th_device=None) -> Pose:
    # return {"position" : th.tensor([x,y,z], device=th_device),
    #         "orientation_xyzw" : th.tensor([qx,qy,qz,qw], device=th_device)}
    return Pose(position = th.tensor([x,y,z], device=th_device),
                orientation_xyzw = th.tensor([qx,qy,qz,qw], device=th_device))

@dataclass
class JointStateArray:
    joints_pve : th.Tensor

    def __init__(self, joints_pve : th.Tensor):
        """

        Parameters
        ----------
        joint_dof_pve : th.Tensor
            A tensor in the shape (joints_number, 3). The first dimension represents the joint index,
            then dimension 1 contains position, velocity and effort, in this order.
        """
        self.joints_pve = joints_pve

    @property
    def positions(self):
        return self.joints_pve[:,0]
    
    @property
    def velocities(self):
        return self.joints_pve[:,1]
    
    @property
    def efforts(self):
        return self.joints_pve[:,2]

@dataclass
class JointState:
    position : th.Tensor
    rate : th.Tensor
    effort : th.Tensor
    
    def __init__(self, position : Union[th.Tensor, List[float], float],
                       rate : Union[th.Tensor, List[float], float],
                       effort : Union[th.Tensor, List[float], float]):
        self.position = th.as_tensor(position).view(-1)
        self.rate = th.as_tensor(rate).view(-1)
        self.effort = th.as_tensor(effort).view(-1)

    
@dataclass
class LinkState:
    pose : Pose
    pos_velocity_xyz : th.Tensor
    ang_velocity_xyz : th.Tensor

    def __init__(self, position_xyz : th.Tensor | tuple, orientation_xyzw : th.Tensor | tuple ,
                    pos_com_velocity_xyz : th.Tensor | tuple, ang_velocity_xyz : th.Tensor | tuple):
        if isinstance(position_xyz, tuple):
            position_xyz = th.as_tensor(position_xyz)
        if isinstance(orientation_xyzw, tuple):
            orientation_xyzw = th.as_tensor(orientation_xyzw)
        if isinstance(pos_com_velocity_xyz, tuple):
            pos_com_velocity_xyz = th.as_tensor(pos_com_velocity_xyz)
        if isinstance(ang_velocity_xyz, tuple):
            ang_velocity_xyz = th.as_tensor(ang_velocity_xyz)
        self.pose = build_pose(position_xyz[0],position_xyz[1],position_xyz[2], orientation_xyzw[0],orientation_xyzw[1],orientation_xyzw[2],orientation_xyzw[3])
        self.pos_velocity_xyz = pos_com_velocity_xyz
        self.ang_velocity_xyz = ang_velocity_xyz








    






def evaluatePolicy(env,
                   model,
                   episodes : int,
                   on_ep_done_callback : Callable[[float, int,int],Any] | None = None,
                   predict_func : Optional[Callable[[Any], Tuple[Any,Any]]] = None,
                   progress_bar : bool = False,
                   images_return = None,
                   obs_return = None):
    with th.no_grad():
        if predict_func is None:
            predict_func_ = model.predict
        else:
            predict_func_ = predict_func
        rewards = np.empty((episodes,), dtype = np.float32)
        steps = np.empty((episodes,), dtype = np.int32)
        wallDurations = np.empty((episodes,), dtype = np.float32)
        predictWallDurations = np.empty((episodes,), dtype = np.float32)
        totDuration=0.0
        successes = 0.0
        #frames = []
        #do an average over a bunch of episodes
        if not progress_bar:
            maybe_tqdm = lambda x:x
        else:
            maybe_tqdm = tqdm.tqdm
        for episode in maybe_tqdm(range(0,episodes)):
            frame = 0
            episodeReward = 0
            terminated = False
            truncated = False
            predDurations = []
            t0 = time.monotonic()
            # ggLog.info("Env resetting...")
            obs, info = env.reset()
            # ggLog.info("Env resetted")
            if images_return is not None:
                images_return.append([])
            if obs_return is not None:
                obs_return.append([])
            while not (terminated or truncated):
                t0_pred = time.monotonic()
                # ggLog.info("Predicting")
                if images_return is not None:
                    images_return[-1].append(env.render())
                if obs_return is not None:
                    obs_return[-1].append(obs)
                action, _states = predict_func_(obs)
                predDurations.append(time.monotonic()-t0_pred)
                # ggLog.info("Stepping")
                obs, stepReward, terminated, truncated, info = env.step(action)
                frame+=1
                episodeReward += stepReward
                # ggLog.info(f"Step reward = {stepReward}")
            rewards[episode]=episodeReward
            if "success" in info.keys():
                if info["success"]:
                    ggLog.info(f"Success {successes} ratio = {successes/(episode+1)}")
                    successes += 1
            steps[episode]=frame
            wallDurations[episode]=time.monotonic() - t0
            predictWallDurations[episode]=sum(predDurations)
            if on_ep_done_callback is not None:
                on_ep_done_callback(episodeReward=episodeReward, steps=frame, episode=episode)
            ggLog.debug("Episode "+str(episode)+" lasted "+str(frame)+" frames, total reward = "+str(episodeReward))
        eval_results = {"reward_mean" : np.mean(rewards),
                        "reward_std" : np.std(rewards),
                        "steps_mean" : np.mean(steps),
                        "steps_std" : np.std(steps),
                        "success_ratio" : successes/episodes,
                        "wall_duration_mean" : np.mean(wallDurations),
                        "wall_duration_std" : np.std(wallDurations),
                        "predict_wall_duration_mean" : np.mean(predictWallDurations),
                        "predict_wall_duration_std" : np.std(predictWallDurations)}
    return eval_results

# from rreal.algorithms.rl_agent import RLAgent
def evaluatePolicyVec(vec_env : gym.vector.VectorEnv,
                   model : "RLAgent | None",
                   episodes : int,
                   on_ep_done_callback : Callable[[float, int,int],Any] | None = None,
                   predict_func : Optional[Callable[[Any, bool], Tuple[Any,Any]]] = None,
                   progress_bar : bool = False,
                   images_return = None,
                   obs_return = None,
                   extra_info_stats : list[str] = [],
                   deterministic : bool = False) -> Dict[str, float]:
    with th.no_grad():
        if model is not None:
            is_training = model.training
            model.eval()
            model_device = model.input_device()
            from adarl.utils.tensor_trees import map_tensor_tree
            def predict_func_(obs, deterministic : bool = False):
                obs = map_tensor_tree(obs, lambda leaf: th.as_tensor(leaf, device=model_device))
                return model.predict(obs, deterministic=deterministic)
        elif predict_func is not None:
            predict_func_ = predict_func
        else:
            raise AttributeError(f"You must set either model or predict_func")
        num_envs  = vec_env.unwrapped.num_envs
        buffsizes = episodes+num_envs # may collect at most num_env excess episodes
        rewards = np.empty((buffsizes,), dtype = np.float32)
        durations_steps = np.empty((buffsizes,), dtype = np.int32)
        extra_stats = {k:np.empty((buffsizes,), dtype = np.float32) for k in extra_info_stats}
        successes = np.zeros((buffsizes,), dtype = np.int32)
        collected_eps = 0
        collected_steps = 0
        used_num_envs = math.gcd(episodes, num_envs)
        if used_num_envs < num_envs:
            ggLog.warn(f"evaluatePolicyVec: Using only {used_num_envs} envs out of {num_envs} to avoid bias in episode statistics (eval episodes={episodes})")
        #frames = []
        #do an average over a bunch of episodes
        if not progress_bar:
            maybe_tqdm = lambda x:x
        else:
            maybe_tqdm = tqdm.tqdm

        running_rews = [0] * used_num_envs
        running_durations = [0] * used_num_envs
        if obs_return is not None:
            running_obss = [[] for i in range(used_num_envs)]
        t0 = time.monotonic()
        tot_step_time = 0.0
        tot_pred_time = 0.0
        term_count = 0
        trunc_count = 0
        obss, infos = vec_env.reset()
        while collected_eps < episodes:
            ts0 = time.monotonic()
            acts, _states = predict_func_(obss, deterministic = deterministic)
            ts1 = time.monotonic()
            obss, rews, terms, truncs, infos = vec_env.step(acts)
            ts2 = time.monotonic()
            collected_steps += used_num_envs
            # ggLog.info(f"Eval: collected steps = {collected_steps}, collected eps = {collected_eps}")
            for i in range(used_num_envs):
                running_rews[i] += rews[i]
                running_durations[i] += 1
                if obs_return is not None:
                    running_obss[i].append(obss[i])
                if terms[i] or truncs[i]:
                    if terms[i]:
                        term_count+=1
                    if truncs[i]:
                        trunc_count+=1
                    tot_reward = running_rews[i].sum()
                    rewards[collected_eps] = tot_reward
                    durations_steps[collected_eps] = running_durations[i]
                    for k in extra_stats:
                        extra_stats[k][collected_eps] = infos[k][i]
                    if obs_return is not None:
                        obs_return.append(running_obss[i])
                    if on_ep_done_callback is not None:
                        on_ep_done_callback(episodeReward=tot_reward, steps=running_durations[i], episode=collected_eps)
                    if "success" in infos.keys():
                        successes[collected_eps] = 1 if infos["success"][i] else 0
                    running_durations[i] = 0
                    running_rews[i] = 0
                    if obs_return is not None:
                        running_obss[i] = []
                    collected_eps += 1
            ts3 = time.monotonic()
            tot_step_time += ts2 - ts1
            tot_pred_time += ts1 - ts0

        tf = time.monotonic()
        eval_results = {"reward_mean" : np.mean(rewards[:episodes]),
                        "reward_std" : np.std(rewards[:episodes]),
                        "steps_mean" : np.mean(durations_steps[:episodes]),
                        "steps_std" : np.std(durations_steps[:episodes]),
                        "success_ratio" : np.sum(successes[:episodes])/episodes,
                        "fps" : collected_steps/(tf-t0),
                        "collected_steps" : collected_steps,
                        "collected_episodes" : collected_eps,
                        "avg_pred_time" : tot_pred_time/(collected_steps/used_num_envs),
                        "avg_step_time" : tot_step_time/(collected_steps/used_num_envs),
                        "terminal_count" : term_count,
                        "truncation_count" : trunc_count}
        eval_results.update({f"{k}_mean":np.mean(v[:episodes]) for k,v in extra_stats.items()})
        eval_results.update({f"{k}_std":np.std(v[:episodes]) for k,v in extra_stats.items()})
        if model is not None:
            model.train(is_training)
    return eval_results


def pyTorch_makeDeterministic(seed):
    """ Make pytorch as deterministic as possible.
        Still, DOES NOT ENSURE REPRODUCIBILTY ACROSS DIFFERENT TORCH/CUDA BUILDS AND
        HARDWARE ARCHITECTURES
    """
    import torch as th
    import random
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = True

    # print(f"Seed set to {seed}")
    # time.sleep(10)
    th.backends.cudnn.benchmark = False
    th.use_deterministic_algorithms(True)
    # Following may make things better, see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"



def list_gpus():
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []

        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            name = pynvml.nvmlDeviceGetName(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)

            # CUDA support
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                cuda_supported = True
                compute_capability = f"{major}.{minor}"
            except pynvml.NVMLError:
                cuda_supported = False
                compute_capability = None

            # VRAM
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_vram = mem.total          # bytes
            free_vram = mem.free            # bytes
            used_vram = mem.used            # bytes
            pci_bus_id = pynvml.nvmlDeviceGetPciInfo(handle).busId

            gpus.append({
                "index": i,
                "name": name,
                "uuid": uuid,
                "cuda_supported": cuda_supported,
                "compute_capability": compute_capability,
                "total_vram": total_vram,
                "free_vram": free_vram,
                "used_vram": used_vram,
                "pci_bus_id": pci_bus_id
            })

        return gpus
    except Exception as e:
        ggLog.warn("Cannot list GPUs")
        return []

def get_gpu_names():
    return [gpu['name'] for gpu in list_gpus()]

def getBestGpu(seed ):
    import torch as th
    gpus_mem_info = []
    for i in range(th.cuda.device_count()):
        prevDev = th.cuda.current_device()
        th.cuda.set_device(th.device(type="cuda", index=i))
        gpus_mem_info.append(th.cuda.mem_get_info()) #Returns [free, total]
        th.cuda.set_device(prevDev)
        # print(f"Got {gpus_mem_info[-1]}")
    gpu_infos = list_gpus()

    bestRatio = 0
    ratios = [0.0]*len(gpu_infos)
    for i in range(len(gpu_infos)):
        tot = gpu_infos[i]['total_vram']
        free = gpu_infos[i]['free_vram']
        ratio = free/tot
        ratios[i] = ratio
        if ratio > bestRatio:
            bestRatio = ratio

    # Look for the gpus that are within 10% of the best one
    candidates = []
    for i in range(len(gpu_infos)):
        if ratios[i] - bestRatio < 0.1:
            candidates.append(i)
    
    chosen_one = candidates[seed%len(candidates)]

    ggLog.info(f"Choosing GPU {chosen_one} with {ratios[chosen_one]*100}% free memory. Candidates were {[f'{i}:{ratios[i]*100}%' for i in candidates]}, seed was {seed}")
    return chosen_one


def torch_selectBestGpu(seed = 0):
    import torch as th
    bestGpu = getBestGpu(seed = seed)
    th.cuda.set_device(bestGpu)
    return th.device('cuda:'+str(bestGpu))


def obs_to_tensor(obs) -> Union[th.Tensor, Dict[Any, th.Tensor]]:
    if isinstance(obs, dict):
        return {k:obs_to_tensor(v) for k,v in obs.items()}
    else:
        return th.as_tensor(obs)



def imgToCvIntRgb(img_chw_rgb : Union[th.Tensor, np.ndarray], min_val = -1, max_val = 1) -> np.ndarray:
    if isinstance(img_chw_rgb, np.ndarray):
        imgTorch = th.as_tensor(img_chw_rgb)
    else:
        imgTorch = img_chw_rgb
    if len(imgTorch.size())==2:
        imgTorch = imgTorch.unsqueeze(0)
    if imgTorch.size()[0] not in [1,3,4]:
        imgTorch = imgTorch.permute(2,0,1) # hwc to chw

    channels = imgTorch.size()[0]
    if channels == 1:
        imgTorch = imgTorch.repeat((3,1,1))
    elif channels == 3:
        imgTorch = imgTorch
    else:
        raise AttributeError(f"Unsupported image shape {imgTorch.size()}")
    
    if imgTorch.dtype in (th.float32, th.float64):
        imgTorch = (imgTorch + (-min_val))/(max_val-min_val) * 255
        imgTorch = imgTorch.to(dtype=th.uint8)
    elif imgTorch.dtype == th.uint8:
        pass
    else:
        raise AttributeError(f"Unsupported image dtype {imgTorch.dtype}")

    imgTorch = imgTorch[[2,1,0]] # rgb to bgr
    imgTorch = imgTorch.permute(1,2,0)
    imgCv = imgTorch.cpu().numpy()
    return imgCv



def randn_like(t : th.Tensor, mu : th.Tensor, std : th.Tensor, generator  : th.Generator):
    return th.randn(size=t.size(),
                    generator=generator,
                    dtype=t.dtype,
                    device=t.device)*std + mu

def randn_from_mustd(mu_std : th.Tensor, generator  : th.Generator | None,
                     squash_sigma : float = -1.0,
                     size : Sequence[int] | None = None):    
    if th.compiler.is_compiling():
        generator = None
    if size is None:
        size = mu_std[0].size()
    noise = th.empty(size=size, dtype=mu_std.dtype, device=mu_std.device).normal_(generator=generator)*mu_std[1] + mu_std[0]
    # noise =  th.randn(size=size,
    #                 generator=generator,
    #                 dtype=mu_std.dtype,
    #                 device=mu_std.device)
    if squash_sigma > 0:
        if squash_sigma < 1.5:
            ggLog.warn(f"Using randn squashing with squash_sigma={squash_sigma}. This may lead to a non-concave distribution!")
        noise = th.tanh(noise/(squash_sigma))*squash_sigma
    return noise*mu_std[1] + mu_std[0]

def to_string_tensor(strings : list[str] | np.ndarray, max_string_len : int = 32):
    return th.as_tensor([list(n.encode("utf-8").ljust(max_string_len)[:max_string_len]) for n in strings], dtype=th.uint8) # ugly, but simple


def pretty_print_tensor_map(thmap : Mapping[str,th.Tensor]):
    n = "\n"
    return n.join([f"{k}:{v.cpu().tolist() if v.numel()<100 else v}" for k,v in thmap.items()])


def hash_tensor(tensor):
    return hash(tuple(tensor.reshape(-1).tolist()))

def conditioned_assign(original : th.Tensor, do_copy : th.Tensor, newvalues : th.Tensor | float | int):
    """Copy newvalues into original only if do_copy is True

    Parameters
    ----------
    original : th.Tensor
        _description_
    do_copy : th.Tensor
        _description_
    newvalues : th.Tensor | float | int
        _description_
    """
    masked_assign(original.unsqueeze(0), do_copy.view(-1), newvalues)

def expand_tensor_into_lower_dims(tensor : th.Tensor, target_size : th.Size) -> th.Tensor:
    # Expand the tensor in the (reversed) upper dimensions
    tensor = tensor.expand(target_size[::-1])
    # Permute the dimensions to restore the required order
    tensor = tensor.permute(*list(range(tensor.ndim - 1, -1, -1))) # using torch arange brings a tensor-list conversion and dynamo is not happy with it
    return tensor

def masked_assign(original : th.Tensor, row_mask : th.Tensor, newvalues : th.Tensor | float | int | bool, inplace : bool = True):
    """Inplace assign values to the original tensor, in locations defined by mask.
        newvalues must have the same shape as original.
        Should equivalent to:
            original[row_mask] = newvalues[row_mask]
    Parameters
    ----------
    original : th.Tensor
        _description_
    mask : th.Tensor
        _description_
    newvalues : th.Tensor
        _description_
    """
    if not isinstance(newvalues, th.Tensor):
        newvalues = th.as_tensor(newvalues)
    # ggLog.info(f"mask.size() = {row_mask.size()}")
    # ggLog.info(f"newvalues.size() = {newvalues.size()}")
    # ggLog.info(f"moriginalask.size() = {original.size()}")
    if len(row_mask.size()) != 1 or row_mask.size()[0] != original.size()[0]:
        raise RuntimeError(f"row_mask must be of size ({(original.size()[0],)}), but it is {row_mask.size()}")
    # mask = row_mask.expand(original.size()[::-1]).T # expand the row mask into lower dimension (like a reverse broadcast)
    mask = expand_tensor_into_lower_dims(row_mask, original.size())
    if inplace:
        th.where(mask,
                newvalues.to(device=original.device, non_blocking=original.device.type == "cuda"), # nonblocking is unsafe for transfers to cpu
                original,
                out=original)
        return original
    else:
        return th.where(mask, newvalues, original)

def masked_assign_sc(original : th.Tensor, mask : th.Tensor, newvalues : th.Tensor | float | int):
    """Inplace assign values to the original tensor, in locations defined by mask.
        At the first dimension newvalues must have the same size as thee are True values in mask,
         so it must be that newvalues.size()=(mask.count_nonzero(),)+original.size()[1:]. Or it
        must be broadcastable to it.
        Should equivalent to:
            original[mask] = newvalues
    Parameters
    ----------
    original : th.Tensor
        _description_
    mask : th.Tensor
        _description_
    newvalues : th.Tensor
        _description_
    """
    if not isinstance(newvalues, th.Tensor):
        newvalues = th.as_tensor(newvalues)
    original.masked_scatter_(mask, 
                             newvalues.to(device=original.device, non_blocking=original.device.type == "cuda"))


def move_masked_to_start(tensor : th.Tensor, row_mask : th.Tensor, out : th.Tensor | None = None):
    """Make a tensor where the rows of tensor where row_mask is True are moved to the start.
        This does not incur in CUDA syncs.

    Parameters
    ----------
    tensor : th.Tensor
        tensor to take the rows from
    row_mask : th.Tensor
        Mask defining which rows to move

    Raises
    ------
    RuntimeError
        _description_
    """
    # We create and indexing tensor that says where to place each element of tensor into out
    # So each index 3 of i says in what row of out the row 3 of tensor must go
    # In all the places where row_mask is False, we put -1, so that all those rows are placed in the last element of out,
    # in this way we always put the last element of tensor in the last element of out, which is always correct.
    i = th.where(row_mask, row_mask.cumsum(0)-1, -1)
    if out is None:
        out = th.zeros_like(tensor)
    out.index_put_((i,), tensor)
    return out

def masked_to_masked_assign(dest_tensor : th.Tensor, dest_row_mask : th.Tensor, src_tensor : th.Tensor, src_row_mask : th.Tensor):
    """Inplace assign values from src_tensor to dest_tensor, in locations defined by src_mask and dest_mask.
        The result is the same as doing:
            dest_tensor[dest_mask] = src_tensor[src_mask]
        However this does not incur in CUDA syncs.
        If the number of True values in src_mask is different from the number of True values in dest_mask,
        the extra elements are ignored, following their order along the zero dimension.

    Parameters
    ----------
    dest_tensor : th.Tensor
        _description_
    dest_mask : th.Tensor
        _description_
    src_tensor : th.Tensor
        _description_
    src_mask : th.Tensor
        _description_

    Returns
    -------
    th.Tensor
        The mask indicating which elements where actually set
    """
    dbg_check_size(dest_row_mask, (dest_tensor.size()[0],), "dest_mask must be 1D and have the same size as dest_tensor first dimension")
    dbg_check_size(src_row_mask, (src_tensor.size()[0],),   "src_mask must be 1D and have the same size as src_tensor first dimension")
    reordered_src = move_masked_to_start(src_tensor, src_row_mask) # move the selected rows to the start
    src_elements_count = th.count_nonzero(src_row_mask)
    clamped_dest_mask = th.logical_and(dest_row_mask, dest_row_mask.cumsum(0)<=src_elements_count) # clamp the dest mask to the number of available elements in src
    mask = expand_tensor_into_lower_dims(clamped_dest_mask, dest_tensor.size())
    dest_tensor.masked_scatter_(mask, reordered_src) # Move the selected rows to the destination
    return clamped_dest_mask

_T = TypeVar('_T', float, th.Tensor)

def unnormalize(v : _T, min : _T, max : _T) -> _T:
    return min+(v+1)/2*(max-min)

def normalize(value : _T, min : _T, max : _T):
    return (value + (-min))/(max-min)*2-1




# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
#                                                     GEOMETRY
# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------






@th.jit.script
def vector_projection(v1 : th.Tensor, v2 : th.Tensor, eps : float = 1e-8):
    """Project v1 onto the direction of v2
    """
    # print(f"v1.size() = {v1.size()}")
    # print(f"v2.size() = {v2.size()}")
    # print(f"th.linalg.norm(v2, dim = -1, keepdim=True).size() = {th.linalg.norm(v2, dim = -1, keepdim=True).size()}")
    v2_norm = v2/th.linalg.norm(v2, dim = -1, keepdim=True)
    # print(f"v2_norm.size() = {v2_norm.size()}")
    # print(f"th.linalg.vecdot(v1,v2_norm, dim=-1).size() = {th.linalg.vecdot(v1,v2_norm, dim=-1).size()}")
    return th.linalg.vecdot(v1,v2_norm, dim=-1).unsqueeze(-1)*(v2_norm + eps)

def vectors_angle(v1 : th.Tensor, v2 : th.Tensor, eps : float = 1e-8):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
        angle = arccos( dot(v1, v2) / (||v1||*||v2||) )
    """
    v1_u = v1 / (th.linalg.norm(v1, dim=-1, keepdim=True) + eps)
    v2_u = v2 / (th.linalg.norm(v2, dim=-1, keepdim=True) + eps)
    return th.acos(th.clamp(th.linalg.vecdot(v1_u, v2_u, dim=-1), -1.0, 1.0))


def quaternionDistance(q1 : quaternion.quaternion, q2 : quaternion.quaternion ):
    """ Returns the minimum angle that separates two orientations.
    """
    # q1a = quaternion.as_float_array(q1)
    # q2a = quaternion.as_float_array(q2)
    #
    # return np.arccos(2*np.square(np.inner(q1a,q2a)) - 1)
    return quaternion.rotation_intrinsic_distance(q1,q2)

def buildQuaternion(x,y,z,w):
    return quaternion.quaternion(w,x,y,z)

def quaternion_xyzw_from_rotmat(rotmat : np.ndarray | th.Tensor):
    if isinstance(rotmat, th.Tensor):
        rotmat_np = rotmat.cpu().numpy()
    else:
        rotmat_np = rotmat
    quat_xyzw = quaternion.as_float_array(quaternion.from_rotation_matrix(rotmat_np))[...,[1,2,3,0]]
    if isinstance(rotmat, th.Tensor):
        return rotmat.new(quat_xyzw)
    else:
        return quat_xyzw

def ros_rpy_to_quaternion_xyzw_th(rpy):
    if not isinstance(rpy, th.Tensor):
        rpy = thtens(rpy)
    zero = th.zeros_like(rpy[...,0])
    roll   = th.stack([th.sin(rpy[...,0]/2),    zero,                   zero,                   th.cos(rpy[...,0]/2)], dim=-1)
    pitch  = th.stack([zero,                    th.sin(rpy[...,1]/2),   zero,                   th.cos(rpy[...,1]/2)], dim=-1)
    yaw    = th.stack([zero,                    zero,                   th.sin(rpy[...,2]/2),   th.cos(rpy[...,2]/2)], dim=-1)
    # On fixed axes:
    # First rotate around x (roll)
    # Then rotate around y (pitch)
    # Then rotate around z (yaw)
    r = quat_mul_xyzw(yaw, quat_mul_xyzw(pitch, roll))
    return r

def ros_rpy_to_quaternion_xyzw(rpy):
    q = ros_rpy_to_quaternion_xyzw_th(rpy)
    return q[0].item(), q[1].item(), q[2].item(), q[3].item()

def _pure_axis_quaternion_xyzw_th(angle : th.Tensor, axis : int):
    """Quaternion (xyzw) of a rotation of ``angle`` radians about a single principal axis.

    ``axis`` selects the rotation axis: 0 -> x (roll), 1 -> y (pitch), 2 -> z (yaw).
    The angle is flattened to a 1-D batch, giving an ``(N, 4)`` output. The output is
    preallocated (the two off-axis vector components stay zero) and sin/cos are written
    straight into the axis and w slices, so no intermediate tensors or stack/cat copy are
    allocated.
    """
    if not isinstance(angle, th.Tensor):
        angle = thtens(angle)
    angle = angle.reshape(-1)
    q = th.zeros((*angle.shape, 4), dtype=angle.dtype, device=angle.device)
    half = angle / 2
    th.sin(half, out=q[..., axis])
    th.cos(half, out=q[..., 3])
    return q

def pure_roll_quaternion_xyzw_th(roll : th.Tensor):
    return _pure_axis_quaternion_xyzw_th(roll, 0)

def pure_pitch_quaternion_xyzw_th(pitch : th.Tensor):
    return _pure_axis_quaternion_xyzw_th(pitch, 1)

def pure_yaw_quaternion_xyzw_th(yaw : th.Tensor):
    return _pure_axis_quaternion_xyzw_th(yaw, 2)



def quat_conj_xyzw_np(quaternion_xyzw : np.ndarray | th.Tensor):
    if isinstance(quaternion_xyzw, th.Tensor):
        quaternion_xyzw = quaternion_xyzw.cpu()
    q = quaternion.from_float_array(quaternion_xyzw[...,[3,0,1,2]])
    q = q.conjugate()
    return quaternion.as_float_array(q)[...,[1,2,3,0]]

def quat_rotate_np(vector_xyz : np.ndarray | th.Tensor, quaternion_xyzw : np.ndarray | th.Tensor):
    if isinstance(quaternion_xyzw, th.Tensor):
        quaternion_xyzw = quaternion_xyzw.cpu().numpy()
    if isinstance(vector_xyz, th.Tensor):
        vector_xyz = vector_xyz.cpu().numpy()
    if len(quaternion_xyzw.shape)<2:
        quaternion_xyzw = np.expand_dims(quaternion_xyzw, axis=0)
        nonvec = True
    else:
        nonvec = False
    q = quaternion.from_float_array(quaternion_xyzw[:,[3,0,1,2]])
    qv = quaternion.from_vector_part(vector_xyz)
    r = quaternion.as_vector_part(q*qv*q.conjugate())
    if nonvec:
        return r[0]
    else:
        return r
    


def th_quat_combine(q_applied_first_xyzw : th.Tensor, q_applied_second_xyzw : th.Tensor):
    return quat_mul_xyzw(q_applied_second_xyzw,q_applied_first_xyzw)

def quat_mul_xyzw_np(q1_xyzw : np.ndarray, q2_xyzw : np.ndarray):
    return quat_mul_xyzw(th.as_tensor(q1_xyzw),
                    th_quat_conj(th.as_tensor(q2_xyzw))).cpu().numpy()
# @th.jit.script
def quat_mul_xyzw(q1_xyzw : th.Tensor, q2_xyzw : th.Tensor):
    """Performs a quaternion multiplication, computing, q1*q2, which is equivalent to rotating by q2 and then by q1

    Parameters
    ----------
    q1_xyzw : th.Tensor
        Quaternoin q1
    q2_xyzw : th.Tensor
        Quaternoin q2

    Returns
    -------
    th.Tensor
        Quaternoin q1*q2
    """
    r1 = q1_xyzw[...,3].unsqueeze(-1)
    v1 = q1_xyzw[...,0:3]
    r2 = q2_xyzw[...,3].unsqueeze(-1)
    v2 = q2_xyzw[...,0:3]
    q = th.empty_like(q1_xyzw)
    q[...,3] = r1[...,0]*r2[...,0] - th.linalg.vecdot(v1,v2)
    q[...,0:3] = r1*v2 + r2*v1 + th.linalg.cross(v1,v2, dim=-1)
    return q


@th.jit.script
def th_quat_conj(q_xyzw : th.Tensor) -> th.Tensor:
    """Gives the inverse rotation of q, usually denoted q^-1 or q'. Note that q*q' = 1
    """
    return q_xyzw*th.tensor([-1.0,-1.0,-1.0,1.0]).to(device=q_xyzw.device, non_blocking=q_xyzw.device.type=="cuda")


def th_quat_rotate_py(vector_xyz : th.Tensor, quaternion_xyzw : th.Tensor):
    vector_xyzw = th.cat([vector_xyz, th.zeros_like(vector_xyz[...,0].unsqueeze(-1))], dim=-1)
    return quat_mul_xyzw(quaternion_xyzw, quat_mul_xyzw(vector_xyzw, th_quat_conj(quaternion_xyzw)))[...,0:3]

@th.jit.script
def th_quat_rotate(vector_xyz : th.Tensor, quaternion_xyzw : th.Tensor) -> th.Tensor:
    return th_quat_rotate_py(vector_xyz=vector_xyz, quaternion_xyzw=quaternion_xyzw)


@th.jit.script
def quat_swing_twist_decomposition_xyzw(quat_xyzw : th.Tensor, axis_xyz : th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    """Decomposes the quaternion into a rotation around
    the axis (twist), and a rotation perpendicular to
    the axis (swing).

    Parameters
    ----------
    quat_wxyz : th.Tensor
        Quaternion rotation
    axis_xyz : th.Tensor
        Axis

    Returns
    -------
    Tuple[th.Tensor, th.Tensor]
        swing, twist
    """
    quat_axis = quat_xyzw[0:3]
    twist_xyzw = th.empty(size=(4,), device=quat_axis.device)
    twist_xyzw[0:3] = vector_projection(quat_axis, axis_xyz)
    twist_xyzw[3] = quat_xyzw[3]
    twist_xyzw = twist_xyzw/twist_xyzw.norm()
    swing_xyzw = quat_mul_xyzw(quat_xyzw,th_quat_conj(twist_xyzw))
    return swing_xyzw, twist_xyzw

@th.jit.script
def quat_angle_xyzw(q_xyzw : th.Tensor) -> th.Tensor:
    """Angle of the angle-axis representation of the quaternion

    Parameters
    ----------
    q_wxyz : th.Tensor
        _description_

    Returns
    -------
    th.Tensor
        _description_
    """
    return 2*th.atan2(th.norm(q_xyzw[...,0:3], dim=-1),q_xyzw[...,3])

def orthogonal_vec(v : th.Tensor):
    minvals = th.amin(v, dim = -1)
    # print(f"v.size() = {v.size()}")
    # print(f"minvals.size() = {minvals.size()}")
    minvals_expanded = minvals.unsqueeze(-1).expand_as(v)
    # print(f"minvals_expanded = {minvals_expanded}")
    minvals_locations = v==minvals_expanded
    # print(f"minvals_locations = {minvals_locations}")
    first_minvals_locations = th.logical_and(minvals_locations.cumsum(dim=-1)==1, minvals_locations)
    # print(f"first_minvals = {first_minvals_locations}")
    shortest_axis = first_minvals_locations.to(dtype=v.dtype)
    # shortest_axis[v==minvals.unsqueeze(-1).expand_as(v)] = 1
    # print(f"shortest_axis = {shortest_axis}")
    # print(f"th.min(v, dim = -1) = {minvals}")
    return th.linalg.cross(v,shortest_axis)

def quat_xyzw_between_vecs_py(v1 : th.Tensor, v2 : th.Tensor):
    """Get the quaternion rotation that brings v1 to v2.
        e.g.: th_quat_rotate_py(unit_x, quat_xyzw_between_vecs_py(unit_x, th.as_tensor([-1.0,0,0]))) == tensor([-1.0,0.0,0.0]))
    Parameters
    ----------
    v1 : th.Tensor
        _description_
    v2 : th.Tensor
        _description_
    """
    quats_xyzw = th.zeros(size=v1.size()[:-1]+(4,), device=v1.device, dtype=v1.dtype)
    vdot = th.linalg.vecdot(v1, v2)
    k = th.linalg.norm(v1, dim = -1) * th.linalg.norm(v2, dim = -1)
    if th.compiler.is_compiling():
        quats_xyzw[...,:3] = th.linalg.cross(v1,v2) # th.compile does not like non-contiguous out tensors :(
    else:
        th.linalg.cross(v1,v2, out=quats_xyzw[...,:3])
    quats_xyzw[...,3] = k + vdot
    flipped_vecs = vdot/k==-1
    ortho_quats = th.zeros_like(quats_xyzw)
    ortho_quats[...,:3] = orthogonal_vec(v1) # make quats that are orthogonal to v1 in the xyz components and zero in w
    masked_assign(quats_xyzw.view(-1,4),
                  flipped_vecs.view(-1),
                  ortho_quats.view(-1,4))
    # quats_xyzw[vdot/k==-1,:3] = orthogonal_vec(v1)[vdot/k==-1]
    # quats_xyzw[vdot/k==-1,3] = 0
    # print(f"vdot = {vdot}")
    # print(f"k = {k}")
    # print(f"vdot/k==-1 = {vdot/k==-1}")
    # print(f"quats_xyzw = {quats_xyzw}")
    # print(f"th.norm(quats_xyzw, dim=-1) = {th.norm(quats_xyzw, dim=-1)}")
    # print(f"orthogonal_vec(v1) = {orthogonal_vec(v1)}")
    return quats_xyzw/th.norm(quats_xyzw, dim=-1).unsqueeze(-1)

# @th.jit.script
# def quat_xyzw_between_vecs(v1 : th.Tensor, v2 : th.Tensor):
#     return quat_xyzw_between_vecs_py(v1,v2)

def average_two_quaternions(q0, q1, eps=1e-6, antipodal : th.Tensor | None = None):
    """
    Weighted midpoint of two unit quaternions (= SLERP midpoint = Markley
    average for N=2 in the equal-weight case; weighted nlerp otherwise).

    q0, q1: (..., 4) unit quaternions, consistent component order.
    w0, w1: scalars or (...,) tensors broadcasting over the batch.
    returns: (..., 4) averaged unit quaternion.
    """
    # double-cover guard: flip q1 into q0's hemisphere
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    
    q = q0 + q1
    n = q.norm(dim=-1, keepdim=True)

    # antipodal guard: near-zero norm ⇒ frames ~180° apart ⇒ ambiguous
    is_antipodal = n < eps
    dbg_check(is_antipodal, build_msg=lambda: f"Antipodal quaternions detected in average_two_quaternions.",
              async_assert=True)
    q = torch.where(is_antipodal, q0, q / n.clamp_min(eps))

    # pin sign for gradient continuity, same convention as before
    q = q * torch.sign(q[..., -1:] + (q[..., -1:] == 0))
    return q


import adarl.adapters.BaseAdapter
def getBlocking(getterFunction : Callable, blocking_timeout_sec : float, env_controller : adarl.adapters.BaseAdapter.BaseAdapter, step_duration_sec : float = 0.1) -> Dict[Tuple[str,str],Any]:
    call_time = time.monotonic()
    last_warn_time = call_time
    while True:
        gottenStuff, missingStuff = getterFunction()
        if len(missingStuff)==0:
            return gottenStuff
        else:
            t = time.monotonic()
            if t-call_time >= blocking_timeout_sec:
                raise RequestFailError(message=f"Failed to get data {missingStuff}. Got {gottenStuff}",
                                    partialResult=gottenStuff)
            else:
                if t - last_warn_time > 0.1:
                    last_warn_time = t
                    ggLog.warn(f"Waiting for {missingStuff} since {t-call_time:.2f}s got {gottenStuff.keys()}")
                env_controller.run(step_duration_sec)


def th_compile_ext(copy_outs : bool = False,
                   just_graphit : bool = False,
                   skip_eval_unsafe_warmup : int = 0,
                   skip_eval_unsafe_manual_arg_guard : int = -1,
                   *compile_args, **compile_kwargs):
    """A wrapper for torch.compile that can automatically copy outputs, useful for problematic cudagraphs

    Parameters
    ----------
    copy_outs : bool, optional
        Whether to copy outputs, by default False
    just_graphit : bool, optional
        Use the graphit wrapper instead of torch.compile, which only does graph tracing and does not try to apply any optimization, 
        it's however quite limited. By default False
    skip_eval_unsafe_warmup : int, optional
        If >0, the returned function will be wrapped with skip_eval_unsafe, and the 
        guards will be skipped after skip_eval_unsafe_warmup calls, by default 0

    Returns
    -------
    Callable
        A wrapped version of the original function that is compiled with torch.compile.
    """
    from adarl.utils.tensor_trees import clone_tensor_tree
    # th._dynamo.utils.cmp_log()
    # ggLog.info(f"th.compiler.is_compiling()={th.compiler.is_compiling()}, stacktrace={''.join(traceback.format_stack())}")
    if just_graphit:
        from adarl.utils.torch_graphing import graphit
        disable = compile_kwargs.pop("disable", False)
        return graphit(disable=disable)
    else:
        def compiling_decorator(func):
            if th.compiler.is_compiling():
                # If already compiling, do nothing
                return func
            else:
                compiled_func = th.compile(model=func, *compile_args, **compile_kwargs)
                if skip_eval_unsafe_warmup > 0:
                    compiled_func = wrap_skip_eval_unsafe(compiled_func, warmup_runs=skip_eval_unsafe_warmup, manual_arg_guard=skip_eval_unsafe_manual_arg_guard)
                if copy_outs:
                    def compile_and_clone(*args, **kwargs):
                        outs = compiled_func(*args, **kwargs)
                        return clone_tensor_tree(outs, detach=False)
                    return compile_and_clone
                else:
                    def compile(*args, **kwargs):
                        return compiled_func(*args, **kwargs)                
                    return compile
    return compiling_decorator


_func_calls_counts : dict[tuple[Callable,Any], int] = {}
def wrap_skip_eval_unsafe(func, warmup_runs : int, manual_arg_guard : int = -1):
    """ Wraps the function with skip_eval_unsafe, so that after warmup_runs,
        torch compile guards are skipped.

    Parameters
    ----------
    func : Callable
        function containing the torch compiled call
    warmup_runs : int
        How many times to run the function beforestrating to skip the guards.
    manual_arg_guard : int, optional
        If >=0, the argument at this position will be used as a guard key, so 
        that the calls count will be tracked separately for each different value of this argument. This is useful if
        the function is called with different argument values that should be treated independently (for example self
        when it's a class method).

    Returns
    -------
    Callable
        The wrapped function
    """
    def wrapped(*args, **kwargs):
        if manual_arg_guard >= 0:
            guard_arg = args[manual_arg_guard]
        else:
            guard_arg = None
        calls_count = _func_calls_counts.get((func, guard_arg), 0)
        _func_calls_counts[(func, guard_arg)] = calls_count + 1         
        if calls_count < warmup_runs:
            return func(*args, **kwargs)
        else:
            # print(f"Skipping eval unsafe guards for {func} with guard_arg={guard_arg} after {calls_count} calls")
            with th.compiler.set_stance(skip_guard_eval_unsafe=True):
                return func(*args, **kwargs)
    return wrapped

def get_func_input_args(exclude : list[str] = []) -> dict:
    _, _, _, values_flocals = inspect.getargvalues(inspect.currentframe().f_back) #type: ignore
    values = dict(values_flocals)
    for name in exclude:
        values.pop(name, None)
    return values


def sample_distr(size, distribution : DistributionDefTh, device : th.device, dtype : th.dtype, generator : th.Generator) -> th.Tensor:
    if isinstance(distribution, th.Tensor):
        return distribution.expand(size).clone()
    elif distribution[0] == "uniform":
        low, high = distribution[1] #type: ignore
        return th.rand(size, device=device, dtype=dtype, generator=generator)*(high-low)+low
    elif distribution[0] == "loguniform":
        low, high = distribution[1] #type: ignore
        lowlog = th.log(low)
        highlog = th.log(high)
        return th.exp(th.rand(size, device=device, dtype=dtype, generator=generator)*(highlog-lowlog)+lowlog)
    elif distribution[0] == "normal":
        if len(distribution[1]) == 2:
            mean, std = distribution[1]
            clamp_width = th.tensor(5.0, device=device, dtype=dtype)
        else:
            mean, std, clamp_width = distribution[1] #type: ignore
        return th.clamp(th.randn(size, device=device, dtype=dtype, generator=generator), -clamp_width, clamp_width)*std+mean
    else:
        raise NotImplementedError(f"Unsupported distribution type {distribution[0]}")
    


TensorLike = Union[th.Tensor, float, List[float]]
DistributionDef = Union[Tuple[str,Tuple[TensorLike, ...]], TensorLike]
DistributionDefTh = Union[Tuple[str,Tuple[th.Tensor, ...]], th.Tensor]

class DistributionTh:
    DistributionDef = Union[Tuple[str,Tuple[TensorLike, ...]], TensorLike]

    def __init__(self,  distribution_def : DistributionTh.DistributionDef,
                        device : th.device,
                        dtype : th.dtype,
                        generator : th.Generator | None = None):
        if isinstance(distribution_def, (float,th.Tensor, np.ndarray)) or not isinstance(distribution_def[0], str):
            distribution_def = ("constant", distribution_def) #type: ignore
        distrtype : str = distribution_def[0]
        distrparams : Sequence = distribution_def[1]
        self._device = device
        self._dtype = dtype
        self._type = distrtype
        self._params = tuple(th.as_tensor(p, device=device, dtype=dtype) for p in distrparams)
        self._rng = generator

    def sample(self,    size, 
                        device : th.device | None = None, 
                        dtype : th.dtype | None = None, 
                        generator : th.Generator | None = None) -> th.Tensor:
        if device is None:
            device = self._device
        if dtype is None:
            dtype = self._dtype
        if generator is None:
            generator = self._rng
        if self._type == "constant":
            return self._params[0].expand(size).clone()
        elif self._type == "uniform":
            low, high = self._params
            return th.rand(size, device=device, dtype=dtype, generator=generator)*(high-low)+low
        elif self._type == "loguniform":
            low, high = self._params
            lowlog = th.log(low)
            highlog = th.log(high)
            return th.exp(th.rand(size, device=device, dtype=dtype, generator=generator)*(highlog-lowlog)+lowlog)
        elif self._type == "normal":
            if len(self._params) == 2:
                mean, std = self._params
                clamp_width = th.tensor(5.0, device=device, dtype=dtype)
            else:
                mean, std, clamp_width = self._params #type: ignore
            return th.clamp(th.randn(size, device=device, dtype=dtype, generator=generator), -clamp_width, clamp_width)*std+mean
        else:
            raise NotImplementedError(f"Unsupported distribution type {self._type}")
        
def distr_is_constant(distr : DistributionDef) -> bool:
    if isinstance(distr, (th.Tensor, float, int)):
        return True
    else:
        distr_type = distr[0]
        if isinstance(distr_type, (float, int)):
            return True # Then it must be a list of numbers, thus a constant distribution
        else:
            if distr_type == "uniform" or distr_type == "loguniform":
                low, high = distr[1]
                return th.all(th.as_tensor(low) == th.as_tensor(high)).item()
            elif distr_type == "normal":
                if len(distr[1]) == 2:
                    mean, std = distr[1]
                else:
                    mean, std, _ = distr[1]
                return th.all(th.as_tensor(std) == 0).item()
            else:
                raise NotImplementedError(f"Unsupported distribution type {distr_type}")
    
    
def distr_to_tensor(distr : DistributionDef, size : tuple[int,...] | None = None, device : th.device | None = None,
                    dtype : th.dtype | None = None) -> DistributionDefTh:
    if isinstance(distr, (float, int, th.Tensor)):
        return th.as_tensor(distr, device=device, dtype=dtype)
    else:
        distr_type = distr[0]
        if isinstance(distr_type, str):
            if size is not None:
                distr_params = tuple(thtens(t, device=device, dtype=dtype).expand(size) for t in distr[1])
            else:
                distr_params = tuple(thtens(t, device=device, dtype=dtype) for t in distr[1])            
            return distr_type, distr_params
        else:
            return thtens(distr, device=device, dtype=dtype)

@staticmethod
def sample_distr(size, distribution : DistributionDefTh, device : th.device, dtype : th.dtype, generator : th.Generator) -> th.Tensor:
    if isinstance(distribution, th.Tensor):
        return distribution.expand(size).clone()
    elif distribution[0] == "uniform":
        low, high = distribution[1] #type: ignore
        return th.rand(size, device=device, dtype=dtype, generator=generator)*(high-low)+low
    elif distribution[0] == "loguniform":
        low, high = distribution[1] #type: ignore
        lowlog = th.log(low)
        highlog = th.log(high)
        return th.exp(th.rand(size, device=device, dtype=dtype, generator=generator)*(highlog-lowlog)+lowlog)
    elif distribution[0] == "normal":
        if len(distribution[1]) == 2:
            mean, std = distribution[1]
            clamp_width = thtens(5.0, device=device, dtype=dtype)
        else:
            mean, std, clamp_width = distribution[1] #type: ignore
        return th.clamp(th.randn(size, device=device, dtype=dtype, generator=generator), -clamp_width, clamp_width)*std+mean
    else:
        raise NotImplementedError(f"Unsupported distribution type {distribution[0]}")

KEY = TypeVar('KEY')
def expand_default_dict(d : Mapping[KEY | str, float] | float,
                        keys: Sequence[KEY],
                        default_key: str = "default") -> dict[KEY, float]:
    """ Expand the input into a dict with all the specified key populated. If the input is already a dict,
        it should contain either the joint name as key or a "default" key for default values. If the input
        is a float, it is used for all joints."""
    if isinstance(d, float):
        expanded_d = {k: d for k in keys}
    elif isinstance(d, dict):
        try:
            expanded_d = {k: d.get(k, d[default_key]) for k in keys}
        except KeyError as e:
            raise RuntimeError(f"Missing default value, provided dict is {d}") from e
    else:
        raise RuntimeError(f"Unexpected type: {type(d)}, expected float or dict")
    return expanded_d

import dataclasses
_Struct = TypeVar("_Struct")  # a dict or a dataclass instance
def override_struct(struct1: _Struct, struct2: Any) -> _Struct:
    """Recursively override values in ``struct1`` with those from ``struct2``, in place.

    Both arguments are nested structures made of dicts and/or dataclass instances.
    ``struct2`` acts as a sparse "patch": for every key/field it defines, the
    corresponding entry in ``struct1`` is replaced, while keys absent from ``struct2``
    are left untouched.

    The two structures need not use the same container types at corresponding levels:
    a field that is a dataclass in ``struct1`` may be overridden by a dict in
    ``struct2`` (and vice-versa). Whenever both the existing value and the override
    value are themselves structures, the override is applied recursively, so nested
    dataclasses can be patched with partial dicts.

    Overriding is restricted to keys that already exist in ``struct1``; new fields are
    never created. This catches typos in the override and avoids setting phantom
    attributes on dataclasses that would not be real fields.

    Parameters
    ----------
    struct1 : dict or dataclass instance
        The structure to be mutated. Modified in place.
    struct2 : dict or dataclass instance
        The structure holding the override values.

    Returns
    -------
    dict or dataclass instance
        ``struct1``, after being mutated, returned for convenience.

    Raises
    ------
    KeyError
        If ``struct2`` contains a key/field that does not exist at the corresponding
        level of ``struct1``.
    """
    if struct2 is None:
        return struct1
    
    def is_struct(obj):
        return isinstance(obj, dict) or (dataclasses.is_dataclass(obj) and not isinstance(obj, type))

    def keys_of(obj):
        if isinstance(obj, dict):
            return list(obj.keys())
        return [f.name for f in dataclasses.fields(obj)]

    def has_key(obj, key):
        if isinstance(obj, dict):
            return key in obj
        return any(f.name == key for f in dataclasses.fields(obj))

    def get_value(obj, key):
        if isinstance(obj, dict):
            return obj[key]
        return getattr(obj, key)

    def set_value(obj, key, value):
        if isinstance(obj, dict):
            obj[key] = value
        else:
            setattr(obj, key, value)

    for key in keys_of(struct2):
        new_value = get_value(struct2, key)
        if not has_key(struct1, key):
            raise KeyError(f"override key {key!r} not present in struct1 "
                           f"(type {type(struct1).__name__}); cannot override a non-existing field")
        old_value = get_value(struct1, key)
        if is_struct(old_value) and is_struct(new_value):
            # Recurse so a dict in struct2 can override a dataclass (or dict) in struct1
            override_struct(old_value, new_value)
        else:
            set_value(struct1, key, new_value)
    return struct1

def dataclass2dict(dc):
    """Shallow-convert a dataclass instance to a dict, dataclasses.asdict does a deep conversion and deepcopies everything."""
    return {field.name: getattr(dc, field.name) for field in dataclasses.fields(dc)}