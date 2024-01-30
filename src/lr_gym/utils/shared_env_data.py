import torch as th
import torch.multiprocessing as mp
from lr_gym.utils.spaces import gym_spaces
from lr_gym.utils.utils import numpy_to_torch_dtype_dict, torch_to_numpy_dtype_dict
import ctypes
from typing import Optional, Any

def create_tensor_tree(batch_size : int, space : gym_spaces.Space, share_mem : bool, device : th.device) -> th.Tensor | dict[Any, th.Tensor]:
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

class SimpleCommander():
    def __init__(self, mp_context):
        self._cmd_count = mp_context.Value(ctypes.c_int64, lock=False)
        self._cmd_count.value = -1
        self._command = mp_context.Array(ctypes.c_char, 128, lock = False)
        self._command.value = b""
        self._command_cond = mp_context.Condition()
        self._received_cmds_count = None
        self._last_received_cmd = None

    def wait_command(self, timeout_s = None) -> bytearray | None:
        if self._received_cmds_count is None:
            self._received_cmds_count = ctypes.c_int64(0)
        with self._command_cond:
            got_command = self._command_cond.wait_for(lambda: self._cmd_count.value==self._received_cmds_count.value, timeout=timeout_s)
            self._last_received_cmd = self._command.value
            self._received_cmds_count.value += 1
        if got_command:
            return self._last_received_cmd
        else:
            return None
        
    def set_command(self, command : str):
        with self._command_cond:
            self._command.value = command.encode()
            self._cmd_count.value += 1
            self._command_cond.notify_all()

class SharedData():
    def __init__(self, observation_space : gym_spaces.Space,
                 action_space : gym_spaces.Space,
                 info_space : gym_spaces.Space,
                 n_envs : int,
                 device : th.device):
        self._observation_space = observation_space
        self._action_space = action_space
        self._info_space = info_space
        self._n_envs = n_envs
        self._device = device
        self._obss = None
        self._acts = None
        self._infos : dict[Any,th.Tensor] = None
        self._reset_infos : dict[Any,th.Tensor] = None
        self._reset_obss = None
        self._terms : th.Tensor = None
        self._truncs : th.Tensor = None
        self._rews : th.Tensor = None

    def build_data(self):
        self._obss = create_tensor_tree(n_envs, self._observation_space, True, device=self._device)
        self._acts = create_tensor_tree(n_envs, self._action_space, True, device=self._device)
        self._infos : dict[Any,th.Tensor] = create_tensor_tree(n_envs, self._info_space, True, device=self._device)
        self._reset_infos : dict[Any,th.Tensor] = create_tensor_tree(n_envs, self._info_space, True, device=self._device)
        self._reset_obss = create_tensor_tree(n_envs, self._observation_space, True, device=self._device)
        self._terms : th.Tensor = th.zeros(size=(self._n_envs,), dtype=th.bool).share_memory_()
        self._truncs : th.Tensor = th.zeros(size=(self._n_envs,), dtype=th.bool).share_memory_()
        self._rews : th.Tensor = th.zeros(size=(self._n_envs,), dtype=th.float32).share_memory_()



class SharedEnvData():
    def __init__(self, n_envs : int,
                 timeout_s : float,
                 mp_context = None):
        self._n_envs = n_envs
        self._timeout_s = timeout_s
        if mp_context is None:
            mp_context = mp

        self._n_envs_stepping = mp_context.Value(ctypes.c_int32, lock=False)
        self._n_envs_stepping.value = 0
        self._n_envs_stepping_cond = mp_context.Condition()

        self._actions_filled = mp_context.Value(ctypes.c_int32, lock=False)
        self._actions_filled.value = 0
        self._actions_filled_cond = mp_context.Condition()

        self._shared_data : SharedData = None

    def set_backend_data(self, shared_data : SharedData):
        self._shared_data = shared_data

    def fill_data(self, env_idx, observation, action, terminated, truncated, reward, info, reset_info, reset_observation):
        if observation is not None:
            # print("writing rew")
            self._shared_data._rews[env_idx] = reward
            # print("wrote rew")
            self._shared_data._terms[env_idx] = terminated
            self._shared_data._truncs[env_idx] = truncated
            fill_tensor_tree(env_idx, observation, self._shared_data._obss)
            fill_tensor_tree(env_idx, action, self._shared_data._acts)
            fill_tensor_tree(env_idx, info, self._shared_data._infos)
        if reset_info is not None:
            fill_tensor_tree(env_idx, reset_observation, self._shared_data._reset_obss)
            fill_tensor_tree(env_idx, reset_info, self._shared_data._reset_infos)
        with self._n_envs_stepping_cond: # decrease number of envs to wait for
            self._n_envs_stepping.value -= 1
            self._n_envs_stepping_cond.notify_all()

    def fill_actions(self, action_batch):
        fill_tensor_tree(None, action_batch, self._shared_data._acts)
        with self._actions_filled_cond: # mark actions as available
            self._actions_filled.value = self._n_envs
            self._actions_filled_cond.notify_all()

    def mark_waiting_data(self):
        with self._n_envs_stepping_cond: # mark number of envs that have to add data
            self._n_envs_stepping.value = self._n_envs
            self._n_envs_stepping_cond.notify_all()

    def wait_actions(self):
        with self._actions_filled_cond:
            got_actions = self._actions_filled_cond.wait_for(lambda: self._actions_filled.value>0, timeout=self._timeout_s)
            self._actions_filled.value -= 1
            self._actions_filled_cond.notify_all()
        if not got_actions:
            raise RuntimeError(f"SharedEnvData wait timed out waiting for actions")
        ret =  self._shared_data._acts
        return ret


    def wait_data(self) -> tuple[th.Tensor | dict[Any, th.Tensor],
                                th.Tensor,
                                th.Tensor,
                                th.Tensor,
                                dict[Any, th.Tensor],
                                th.Tensor | dict[Any, th.Tensor],
                                dict[Any, th.Tensor]]:
        with self._n_envs_stepping_cond:
            didnt_timeout = self._n_envs_stepping_cond.wait_for(lambda: self._n_envs_stepping.value<=0, timeout=self._timeout_s)
        if not didnt_timeout:
            raise RuntimeError(f"SharedEnvData wait timed out waiting for steps")
        return (self._shared_data._obss,
                self._shared_data._rews,
                self._shared_data._terms,
                self._shared_data._truncs,
                self._shared_data._infos,
                self._shared_data._reset_obss,
                self._shared_data._reset_infos)















def worker_func(sh : SharedEnvData, sc, worker_id):
    # print(f"worker starting with {sh} and {steps}")
    import time
    i = 0
    while True:
        if i%100 == 0:
            print(f"worker step {i}")
        cmd = sc.wait_command()
        i += 1
        if cmd == b"step":
            with th.no_grad():
                act = sh.wait_actions().detach().clone()
            sh.fill_data(   env_idx = 0,
                            observation={"obs": th.tensor([0.1,0.1])},
                            action = th.tensor([0.01,0.01]),
                            terminated = True,
                            truncated = True,
                            reward = 4,
                            info = {"boh":th.tensor([i,i])},
                            reset_info=None,
                            reset_observation=None)
        elif cmd == b"reset":
            sh.fill_data(   env_idx = 0,
                            observation=None,
                            action = None,
                            terminated = None,
                            truncated = None,
                            reward = None,
                            info = None,
                            reset_info={"boh":th.tensor([1,1])},
                            reset_observation={"obs": th.tensor([1,1])})
        elif cmd == b"close":
            break
        elif cmd == b"set_backend_data":
            sh.set_backend_data

        # print(f"worker step {i} end")

    # print(f"worker finished")

if __name__ == "__main__":
    import time
    import numpy as np
    n_envs = 2
    steps = 100000
    ctx = mp.get_context("forkserver")
    info_example = {"boh":th.tensor([0.0,0])}
    info_space = space_from_tree(info_example)
    print(f"info_space = {info_space}")
    sc = SimpleCommander(ctx)
    sh = SharedEnvData(observation_space=gym_spaces.Dict({"obs":gym_spaces.Box(low = np.array([-1,-1]), high=np.array([1.0,1.0]))}),
                       action_space=gym_spaces.Box(low = np.array([-1,-1]), high=np.array([1.0,1.0])),
                       info_space=info_space,
                    n_envs=n_envs,timeout_s=60, device=th.device("cpu"),
                    mp_context=ctx)
    workers = [ctx.Process(target=worker_func, args=(sh,sc,idx)) for idx in range(n_envs)]
    for worker in workers:
        worker.start()
    t0 = time.monotonic()
    for i in range(steps):
        if i%100 == 0:
            print(f"main step {i}")
        sc.set_command("step")
        sh.mark_waiting_data()
        sh.fill_actions(th.tensor([1,1]))
        data = sh.wait_data()
        # print(f"main step {i} end")
    tf = time.monotonic()
    print(f"main(): took {tf-t0}, {steps/(tf-t0)} fps, {(tf-t0)/steps}s per step")
    print(" # \n".join([str(d) for d in data]))
    sh.mark_waiting_data()
    sc.set_command("reset")
    sh.wait_data()
    print(f"resetted")
    print(" # \n".join([str(d) for d in data]))



    sc.set_command("close")

    worker.join()
    