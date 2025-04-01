

from __future__ import annotations

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
                   deterministic : bool = False):
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
        successes = np.empty((buffsizes,), dtype = np.int32)
        collected_eps = 0
        collected_steps = 0
        #frames = []
        #do an average over a bunch of episodes
        if not progress_bar:
            maybe_tqdm = lambda x:x
        else:
            maybe_tqdm = tqdm.tqdm

        running_rews = [0] * num_envs
        running_durations = [0] * num_envs
        if obs_return is not None:
            running_obss = [[] for i in range(num_envs)]
        t0 = time.monotonic()
        obss, infos = vec_env.reset()
        while collected_eps < episodes:
            acts, _states = predict_func_(obss, deterministic = deterministic)
            obss, rews, terms, truncs, infos = vec_env.step(acts)
            collected_steps += num_envs
            for i in range(num_envs):
                running_rews[i] += rews[i]
                running_durations[i] += 1
                if obs_return is not None:
                    running_obss[i].append(obss[i])
                if terms[i] or truncs[i]:
                    rewards[collected_eps] = running_rews[i]
                    durations_steps[collected_eps] = running_durations[i]
                    for k in extra_stats:
                        extra_stats[k][collected_eps] = infos[k][i]
                    if obs_return is not None:
                        obs_return.append(running_obss[i])
                    if on_ep_done_callback is not None:
                        on_ep_done_callback(episodeReward=running_rews[i], steps=running_durations[i], episode=collected_eps)
                    if "success" in infos.keys():
                        successes[collected_eps] = 1 if infos["success"][i] else 0
                    running_durations[i] = 0
                    running_rews[i] = 0
                    if obs_return is not None:
                        running_obss[i] = []
                    collected_eps += 1
        tf = time.monotonic()
        eval_results = {"reward_mean" : np.mean(rewards[:episodes]),
                        "reward_std" : np.std(rewards[:episodes]),
                        "steps_mean" : np.mean(durations_steps[:episodes]),
                        "steps_std" : np.std(durations_steps[:episodes]),
                        "success_ratio" : sum(successes[:episodes])/episodes,
                        "fps" : collected_steps/(tf-t0),
                        "collected_steps" : collected_steps,
                        "collected_episodes" : collected_eps}
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
    th.manual_seed(seed)
    np.random.seed(seed)
    # print(f"Seed set to {seed}")
    # time.sleep(10)
    th.backends.cudnn.benchmark = False
    th.use_deterministic_algorithms(True)
    # Following may make things better, see https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def getBestGpu(seed ):
    import torch as th
    gpus_mem_info = []
    for i in range(th.cuda.device_count()):
        prevDev = th.cuda.current_device()
        th.cuda.set_device(th.device(type="cuda", index=i))
        gpus_mem_info.append(th.cuda.mem_get_info()) #Returns [free, total]
        th.cuda.set_device(prevDev)
        # print(f"Got {gpus_mem_info[-1]}")

    bestRatio = 0
    bestGpu = None
    ratios = [0.0]*len(gpus_mem_info)
    for i in range(len(gpus_mem_info)):
        tot = gpus_mem_info[i][1]
        free = gpus_mem_info[i][0]
        ratio = free/tot
        ratios[i] = ratio
        if ratio > bestRatio:
            bestRatio = ratio
            bestGpu = i

    # Look for the gpus that are within 10% of the best one
    candidates = []
    for i in range(len(gpus_mem_info)):
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

def randn_from_mustd(mu_std : th.Tensor, generator  : th.Generator,
                     squash_sigma : float = -1.0,
                     size : Sequence[int] | None = None):
    if size is None:
        size = mu_std[0].size()
    noise =  th.randn(size=size,
                    generator=generator,
                    dtype=mu_std.dtype,
                    device=mu_std.device)
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


def masked_assign(original : th.Tensor, row_mask : th.Tensor, newvalues : th.Tensor | float | int):
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
    mask = row_mask.expand(original.size()[::-1]).T # expand the row mask into lower dimension (kinda a reverse broadcast)
    th.where(mask,
             newvalues.to(device=original.device, non_blocking=original.device.type == "cuda"), # nonblocking is unsafe for transfers to cpu
             original,
             out=original)

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
def vector_projection(v1 : th.Tensor, v2 : th.Tensor):
    """Project v1 onto the direction of v2
    """
    # print(f"v1.size() = {v1.size()}")
    # print(f"v2.size() = {v2.size()}")
    # print(f"th.linalg.norm(v2, dim = -1, keepdim=True).size() = {th.linalg.norm(v2, dim = -1, keepdim=True).size()}")
    v2_norm = v2/th.linalg.norm(v2, dim = -1, keepdim=True)
    # print(f"v2_norm.size() = {v2_norm.size()}")
    # print(f"th.linalg.vecdot(v1,v2_norm, dim=-1).size() = {th.linalg.vecdot(v1,v2_norm, dim=-1).size()}")
    return th.linalg.vecdot(v1,v2_norm, dim=-1).unsqueeze(-1)*v2_norm




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
    rpy = th.as_tensor(rpy)
    roll  = th.as_tensor([th.sin(rpy[0]/2),  0.0,               0.0,                th.cos(rpy[0]/2)], device=rpy.device)
    pitch = th.as_tensor([0.0,               th.sin(rpy[1]/2),  0.0,                th.cos(rpy[1]/2)], device=rpy.device)
    yaw   = th.as_tensor([0.0,               0.0,               th.sin(rpy[2]/2),   th.cos(rpy[2]/2)], device=rpy.device)
    # On fixed axes:
    # First rotate around x (roll)
    # Then rotate around y (pitch)
    # Then rotate around z (yaw)
    return quat_mul_xyzw(yaw, quat_mul_xyzw(pitch, roll))

def ros_rpy_to_quaternion_xyzw(rpy):
    q = ros_rpy_to_quaternion_xyzw_th(rpy)
    return q[0].item(), q[1].item(), q[2].item(), q[3].item()



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
def th_quat_rotate(vector_xyz : th.Tensor, quaternion_xyzw : th.Tensor):
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
    shortest_axis = th.zeros_like(v)
    minvals = th.min(v, dim = -1)[0]
    # print(f"v.size() = {v.size()}")
    # print(f"minvals.size() = {minvals.size()}")
    shortest_axis[v==minvals.unsqueeze(-1).expand_as(v)] = 1
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
    th.linalg.cross(v1,v2, out=quats_xyzw[...,:3])
    quats_xyzw[...,3] = k + vdot
    quats_xyz = quats_xyzw[...,:3]
    quats_w = quats_xyzw[...,3]
    flipped_vecs = vdot/k==-1
    masked_assign(quats_xyz.view(-1,3), flipped_vecs.view(-1), orthogonal_vec(v1).view(-1,3))
    masked_assign(quats_w.view(-1,1), flipped_vecs.view(-1), 0)
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
