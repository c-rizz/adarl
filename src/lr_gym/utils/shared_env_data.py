import torch as th
import torch.multiprocessing as mp
from lr_gym.utils.spaces import gym_spaces
from lr_gym.utils.utils import numpy_to_torch_dtype_dict, torch_to_numpy_dtype_dict
import ctypes
from typing import Optional, Any
import numpy as np

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
        raise RuntimeError(f"Unexpected tree element type {type(src_tree)}")

def map_tensor_tree(src_tree : dict | th.Tensor, func):
    if isinstance(src_tree, dict):
        r = {}
        for k in src_tree.keys():
            r[k] = map_tensor_tree(src_tree[k], func = func)
        return r
    else:
        return func(src_tree)

def unstack_tensor_tree(src_tree : dict | th.Tensor) -> list:
    if isinstance(src_tree, dict):
        # print(f"src_tree = {src_tree}")
        dictlist = None
        for k in src_tree.keys():
            unstacked_subtree = unstack_tensor_tree(src_tree[k])
            stack_size = len(unstacked_subtree)
            if dictlist is None:
                dictlist = [{} for _ in range(stack_size)]
            for i in range(stack_size):
                dictlist[i][k] = unstacked_subtree[i]
        # print(f"src_tree = {src_tree}, unstacked = {dictlist}")
        return dictlist
    elif isinstance(src_tree, th.Tensor):
        ret = src_tree.unbind()
        # print(f"src_tree = {src_tree}, unstacked = {ret}")
        return ret
    else:
        raise RuntimeError(f"Unexpected tree element type {type(src_tree)}")

def space_from_tree(tensor_tree):
    if isinstance(tensor_tree, dict):
        subspaces = {}
        for k in tensor_tree.keys():
            subspaces[k] = space_from_tree(tensor_tree[k])
        return gym_spaces.Dict(subspaces)
    if isinstance(tensor_tree, np.ndarray):
        tensor_tree = th.as_tensor(tensor_tree)
    if isinstance(tensor_tree, (float, int)):
        tensor_tree = th.as_tensor(tensor_tree)
    if isinstance(tensor_tree, th.Tensor):
        return gym_spaces.Box(high=(th.ones_like(tensor_tree)*float("+inf")).cpu().numpy(),
                              low=(th.ones_like(tensor_tree)*float("-inf")).cpu().numpy(),
                              dtype=torch_to_numpy_dtype_dict[tensor_tree.dtype])
    else:
        raise RuntimeError(f"Unexpected tree element type {tensor_tree}")

class SimpleCommander():
    def __init__(self, mp_context, n_envs, timeout_s):
        self._n_envs = n_envs
        self._cmds_sent_count = mp_context.Value(ctypes.c_uint64, lock=False)
        self._cmds_sent_count.value = 0
        self._current_command = mp_context.Array(ctypes.c_char, 128, lock = False)
        self._current_command.value = b""
        self._cmds_done_count = mp_context.Value(ctypes.c_uint64, lock=False)
        self._cmds_done_count.value = 0
        self._new_cmd_cond = mp_context.Condition()
        self._cmd_done_cond = mp_context.Condition()
        self._received_cmds_count = None # non-shared received commands counter
        self._last_received_cmd = None
        self._timeout_s = timeout_s

    def wait_command(self) -> bytearray | None:
        if self._received_cmds_count is None:
            self._received_cmds_count = ctypes.c_uint64(1)
        with self._new_cmd_cond:
            # print(f"waiting for command {self._received_cmds_count}")
            got_command = self._new_cmd_cond.wait_for(lambda: self._cmds_sent_count.value==self._received_cmds_count.value, timeout=self._timeout_s)
            # print(f"got command {self._received_cmds_count}")
            if got_command:
                self._last_received_cmd = self._current_command.value
                self._received_cmds_count.value += 1
        if got_command:
            return self._last_received_cmd
        else:
            return None
        
    def mark_done(self):
        with self._cmd_done_cond:
            self._cmds_done_count.value += 1
            self._cmd_done_cond.notify_all()

    def wait_done(self):
        done = False
        with self._cmd_done_cond:
            done = self._cmd_done_cond.wait_for(lambda: self._cmds_done_count.value==self._cmds_sent_count.value*self._n_envs, timeout=self._timeout_s)
        if not done:      
            raise TimeoutError(f"Timed out waiting for cmd {self._current_command.value} completion")
        
    def set_command(self, command : str):
        with self._new_cmd_cond:
            self._current_command.value = command.encode()
            self._cmds_sent_count.value += 1
            # print(f"Sent command {self._cmds_sent_count}")
            self._new_cmd_cond.notify_all()

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
        self.build_data()

    def build_data(self):
        self._obss = create_tensor_tree(self._n_envs, self._observation_space, True, device=self._device)
        self._acts = create_tensor_tree(self._n_envs, self._action_space, True, device=self._device)
        self._infos : dict[Any,th.Tensor] = create_tensor_tree(self._n_envs, self._info_space, True, device=self._device)
        self._reset_infos : dict[Any,th.Tensor] = create_tensor_tree(self._n_envs, self._info_space, True, device=self._device)
        self._reset_obss = create_tensor_tree(self._n_envs, self._observation_space, True, device=self._device)
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

    def set_data_struct(self, shared_data : SharedData):
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















def worker_func(sh : SharedEnvData, sc, worker_id, receiver):
    # print(f"worker starting with {sh} and {steps}")
    import time
    i = 0
    running = True
    while running:
        if i%100 == 0:
            print(f"worker step {i}")
        # print(f"waiting command...")
        cmd = sc.wait_command()
        # print(f"got command {cmd}")
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
            running = False
        elif cmd == b"set_backend_data":
            # print(f"[{worker_id}] setting backend data")
            data : SharedData = receiver.recv()
            sh.set_data_struct(data)
            # print(f"[{worker_id}] set backend data")
        if cmd is not None:
            sc.mark_done()


        # print(f"worker step {i} end")
    receiver.close()
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
    sc = SimpleCommander(ctx, n_envs=n_envs)
    sh = SharedEnvData(n_envs=n_envs,timeout_s=5, mp_context=ctx)
    receivers, senders = zip(*[ctx.Pipe(duplex=False) for _ in range(n_envs)])
    workers = [ctx.Process(target=worker_func, args=(sh,sc,idx, receivers[idx])) for idx in range(n_envs)]
    for worker in workers:
        worker.start()

    data = SharedData(observation_space=gym_spaces.Dict({"obs":gym_spaces.Box(low = np.array([-1,-1]), high=np.array([1.0,1.0]))}),
                       action_space=gym_spaces.Box(low = np.array([-1,-1]), high=np.array([1.0,1.0])),
                       info_space=info_space,
                    n_envs=n_envs, device=th.device("cpu"))
    
    sh.set_data_struct(data)
    sc.set_command("set_backend_data")
    for idx in range(n_envs): senders[idx].send(data)
    print("sent backend data")
    sc.wait_done()
    print("backend data set")

    t0 = time.monotonic()
    for i in range(steps):
        if i%100 == 0:
            print(f"main step {i}")
        sc.set_command("step")
        sh.mark_waiting_data()
        sh.fill_actions(th.tensor([1,1]))
        sc.wait_done()

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
    for idx in range(n_envs): senders[idx].close()



    sc.set_command("close")

    worker.join()
    