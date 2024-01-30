import torch as th
import torch.multiprocessing as mp
from lr_gym.utils.spaces import gym_spaces
from lr_gym.utils.utils import numpy_to_torch_dtype_dict, torch_to_numpy_dtype_dict
import ctypes
from typing import Optional

def create_tensor_tree(batch_size : int, space : gym_spaces.Space, share_mem : bool, device : th.device):
    if isinstance(space, gym_spaces.Dict):
        tree = {}
        for k, s in space.spaces.items():
            tree[k] = create_tensor_tree(batch_size, s, share_mem, device)
        return tree
    elif isinstance(space, gym_spaces.Box):
        thdtype = th.as_tensor(space.sample()).dtype
        t = th.zeros(size=(batch_size,)+space.shape, dtype=thdtype, device = device)
        if share_mem:
            t.share_memory_()
        return t
    else:
        raise RuntimeError(f"Unsupported space {space}")

def fill_tensor_tree(env_idx : Optional[int], src_tree : dict | th.Tensor, dst_tree : dict | th.Tensor, depth = 0):
    if isinstance(src_tree, dict):
        if not isinstance(dst_tree,dict):
            raise RuntimeError(f"Tree element type mismatch. src = {type(src_tree)}, dst = {dst_tree}")
        if src_tree.keys() != dst_tree.keys():
            raise RuntimeError(f"source and destination keys do not match: src={src_tree.keys()} dst={dst_tree.keys()}")
        for k in dst_tree.keys():
            fill_tensor_tree(env_idx, src_tree[k], dst_tree[k], depth = depth+1)
        if depth == 0:
            th.cuda.synchronize() # sync non-blocking copies
    elif isinstance(src_tree, th.Tensor):
        if not isinstance(dst_tree,th.Tensor):
            raise RuntimeError(f"Tree element type mismatch. src = {type(src_tree)}, dst = {dst_tree}")
        if env_idx is not None:
            dst_tree[env_idx].copy_(src_tree, non_blocking=True)
        else:
            dst_tree.copy_(src_tree, non_blocking=True)
    else:
        raise RuntimeError(f"Unexpected tree element type {src_tree}")

def space_from_tree(tensor_tree):
    if isinstance(tensor_tree, dict):
        subspaces = {}
        for k in tensor_tree.keys():
            subspaces[k] = space_from_tree(tensor_tree[k])
        return gym_spaces.Dict(subspaces)
    elif isinstance(tensor_tree, th.Tensor):
        return gym_spaces.Box(high=(th.ones_like(tensor_tree)*float("+inf")).cpu().numpy(),
                              low=(th.ones_like(tensor_tree)*float("-inf")).cpu().numpy(),
                              dtype=torch_to_numpy_dtype_dict[tensor_tree.dtype])
    else:
        raise RuntimeError(f"Unexpected tree element type {tensor_tree}")

class SharedEnvData():
    def __init__(self, observation_space : gym_spaces.Space,
                 action_space : gym_spaces.Space,
                 info_space : gym_spaces.Space,
                 n_envs : int,
                 timeout_s : float,
                 device : th.device,
                 mp_context = None):
        self._observation_space = observation_space
        self._action_space = action_space
        self._info_space = info_space
        self._n_envs = n_envs
        self._timeout_s = timeout_s
        if mp_context is None:
            mp_context = mp

        self._shared_obss = create_tensor_tree(n_envs, self._observation_space, True, device=device)
        self._shared_acts = create_tensor_tree(n_envs, self._action_space, True, device=device)
        self._shared_infs = create_tensor_tree(n_envs, self._info_space, True, device=device)
        self._shared_dones = th.zeros(size=(self._n_envs,), dtype=th.bool).share_memory_()
        self._shared_truncs = th.zeros(size=(self._n_envs,), dtype=th.bool).share_memory_()
        self._shared_rews = th.zeros(size=(self._n_envs,), dtype=th.float32).share_memory_()

        self._n_envs_stepping = mp_context.Value(ctypes.c_int32, lock=False)
        self._n_envs_stepping.value = 0
        self._n_envs_stepping_cond = mp_context.Condition()

        self._actions_filled = mp_context.Value(ctypes.c_int32, lock=False)
        self._actions_filled.value = 0
        self._actions_filled_cond = mp_context.Condition()



    def fill_step_data(self, env_idx, observation, action, done, truncated, reward, info):
        # print("writing rew")
        self._shared_rews[env_idx] = reward
        # print("wrote rew")
        self._shared_dones[env_idx] = done
        self._shared_truncs[env_idx] = truncated
        fill_tensor_tree(env_idx, observation, self._shared_obss)
        fill_tensor_tree(env_idx, action, self._shared_acts)
        fill_tensor_tree(env_idx, info, self._shared_infs)
        with self._n_envs_stepping_cond: # decrease number of envs to wait for
            self._n_envs_stepping.value -= 1
            self._n_envs_stepping_cond.notify_all()

    def fill_actions(self, action_batch):
        fill_tensor_tree(None, action_batch, self._shared_acts)
        with self._n_envs_stepping_cond: # mark number of envs that have to step
            self._n_envs_stepping.value = self._n_envs
            self._n_envs_stepping_cond.notify_all()
        with self._actions_filled_cond: # mark actions as available
            self._actions_filled.value = self._n_envs
            self._actions_filled_cond.notify_all()


    def wait_actions(self):
        with self._actions_filled_cond:
            got_actions = self._actions_filled_cond.wait_for(lambda: self._actions_filled.value>0, timeout=self._timeout_s)
            self._actions_filled.value -= 1
            self._actions_filled_cond.notify_all()
        if not got_actions:
            raise RuntimeError(f"SharedEnvData wait timed out waiting for actions")
        ret =  self._shared_acts
        return ret


    def wait_steps(self):
        with self._n_envs_stepping_cond:
            didnt_timeout = self._n_envs_stepping_cond.wait_for(lambda: self._n_envs_stepping.value<=0, timeout=self._timeout_s)
        if not didnt_timeout:
            raise RuntimeError(f"SharedEnvData wait timed out waiting for steps")















def worker_func(sh, steps):
    # print(f"worker starting with {sh} and {steps}")
    import time
    for i in range(steps):
        # print(f"worker step {i}")
        with th.no_grad():
            act = sh.wait_actions().detach().clone()
        sh.fill_step_data(   env_idx = 0,
                        observation={"obs": th.tensor([0.1,0.1])},
                        action = th.tensor([0.01,0.01]),
                        done = True,
                        truncated = True,
                        reward = 4,
                        info = {"boh":th.tensor([i,i])})
        # print(f"worker step {i} end")

    # print(f"worker finished")

if __name__ == "__main__":
    import time
    import numpy as np
    steps = 100000
    ctx = mp.get_context("forkserver")
    info_example = {"boh":th.tensor([0.0,0])}
    info_space = space_from_tree(info_example)
    print(f"info_space = {info_space}")
    sh = SharedEnvData(observation_space=gym_spaces.Dict({"obs":gym_spaces.Box(low = np.array([-1,-1]), high=np.array([1.0,1.0]))}),
                       action_space=gym_spaces.Box(low = np.array([-1,-1]), high=np.array([1.0,1.0])),
                       info_space=info_space,
                    n_envs=1,timeout_s=60, device=th.device("cpu"),
                    mp_context=ctx)
    worker = ctx.Process(target=worker_func, args=(sh,steps))
    worker.start()
    t0 = time.monotonic()
    for i in range(steps):
        # print(f"main step {i}")
        sh.fill_actions(th.tensor([1,1]))
        sh.wait_steps()
        # print(f"main step {i} end")
    tf = time.monotonic()
    print(f"main(): took {tf-t0}, {steps/(tf-t0)} fps, {(tf-t0)/steps}s per step")
    print(f" acts = {sh._shared_acts}")
    print(f" obss = {sh._shared_obss}")
    print(f" infos = {sh._shared_infs}")
    print(f" rews = {sh._shared_rews}")
    worker.join()
    