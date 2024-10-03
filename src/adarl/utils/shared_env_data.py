from __future__ import annotations
import torch as th
import torch.multiprocessing as mp
from adarl.utils.spaces import gym_spaces, ThBox
from adarl.utils.utils import numpy_to_torch_dtype_dict, torch_to_numpy_dtype_dict
import ctypes
from typing import Optional, Any, Tuple, Dict
import numpy as np
from adarl.utils.tensor_trees import create_tensor_tree, fill_tensor_tree, space_from_tree, TensorTree, is_all_finite, map_tensor_tree
import adarl.utils.mp_helper as mp_helper

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
            didnt_timeout = self._new_cmd_cond.wait_for(lambda: self._cmds_sent_count.value==self._received_cmds_count.value, timeout=self._timeout_s)
            # print(f"got command {self._received_cmds_count}")
            if didnt_timeout:
                self._last_received_cmd = self._current_command.value
                self._received_cmds_count.value += 1
        if didnt_timeout:
            return self._last_received_cmd
        else:
            return None
        
    def mark_done(self):
        with self._cmd_done_cond:
            self._cmds_done_count.value += 1
            self._cmd_done_cond.notify_all()

    def wait_done(self, timeout = None):
        done = False
        if timeout is None:
            timeout = self._timeout_s
        with self._cmd_done_cond:
            done = self._cmd_done_cond.wait_for(lambda: self._cmds_done_count.value==self._cmds_sent_count.value*self._n_envs, timeout=timeout)
        if not done:      
            raise TimeoutError(f"Timed out waiting for cmd {self._current_command.value} completion (timeout = {timeout})")
        
    def set_command(self, command : str):
        with self._new_cmd_cond:
            self._current_command.value = command.encode()
            self._cmds_sent_count.value += 1
            # print(f"Sent command {self._cmds_sent_count}")
            self._new_cmd_cond.notify_all()

    def current_command(self) -> str | None:
        with self._cmd_done_cond:
            done = self._cmds_done_count.value==self._cmds_sent_count.value*self._n_envs
            if not done:
                cmd = self._current_command.value.decode("utf-8")
            else:
                cmd = None
        return cmd

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
        # self._consequent_observations = None
        # self._actions = None
        # self._consequent_infos : Dict[Any,th.Tensor] = None
        # self._next_start_infos : Dict[Any,th.Tensor] = None
        # self._next_start_observations = None
        # self._terminations : th.Tensor = None
        # self._truncations : th.Tensor = None
        # self._rewards : th.Tensor = None
        self.build_data()

    def observation_space(self):
        return self._observation_space

    def action_space(self):
        return self._action_space

    def info_space(self):
        return self._info_space

    def build_data(self):
        self._consequent_observations = create_tensor_tree(self._n_envs, self._observation_space, share_mem=True, device=self._device)
        self._actions =                 create_tensor_tree(self._n_envs, self._action_space, share_mem=True, device=self._device)
        self._consequent_infos =        create_tensor_tree(self._n_envs, self._info_space, share_mem=True, device=self._device)
        self._next_start_infos =        create_tensor_tree(self._n_envs, self._info_space, share_mem=True, device=self._device)
        self._next_start_observations = create_tensor_tree(self._n_envs, self._observation_space, share_mem=True, device=self._device)
        self._terminations : th.Tensor =    th.zeros(size=(self._n_envs,), dtype=th.bool, device=self._device).share_memory_()
        self._truncations : th.Tensor =     th.zeros(size=(self._n_envs,), dtype=th.bool, device=self._device).share_memory_()
        self._rewards : th.Tensor =         th.zeros(size=(self._n_envs,), dtype=th.float32, device=self._device).share_memory_()



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

    def data(self):
        return self._shared_data

    def set_data_struct(self, shared_data : SharedData):
        self._shared_data = shared_data

    def fill_data(self, env_idx,
                        next_start_info,
                        next_start_observation,
                        action,
                        terminated,
                        truncated,
                        reward,
                        consequent_observation,
                        consequent_info):
        if consequent_observation is not None:
            # print("writing rew")
            self._shared_data._rewards[env_idx].copy_(reward, non_blocking=True)
            # print("wrote rew")
            self._shared_data._terminations[env_idx].copy_(terminated, non_blocking=True)
            self._shared_data._truncations[env_idx].copy_(truncated, non_blocking=True)
            fill_tensor_tree(env_idx, consequent_observation, self._shared_data._consequent_observations, non_blocking=True)
            fill_tensor_tree(env_idx, action, self._shared_data._actions, non_blocking=True)
            fill_tensor_tree(env_idx, consequent_info, self._shared_data._consequent_infos, nonstrict=True, non_blocking=True) # Implement some mechanism to add missing keys to sharde_data.infos
        if next_start_info is not None:
            fill_tensor_tree(env_idx, next_start_observation, self._shared_data._next_start_observations, non_blocking=True)
            fill_tensor_tree(env_idx, next_start_info, self._shared_data._next_start_infos, nonstrict=True, non_blocking=True) # Implement some mechanism to add missing keys to sharde_data.infos
        if th.cuda.is_initialized(): # synchronize always unless cuda has never even been used (this way we save a lot of memory)
            th.cuda.synchronize()
        with self._n_envs_stepping_cond: # decrease number of envs to wait for
            self._n_envs_stepping.value -= 1
            self._n_envs_stepping_cond.notify_all()


    def fill_actions(self, action_batch):
        if not isinstance(action_batch, th.Tensor):
            action_batch = th.as_tensor(action_batch, device=self._shared_data._device)
        fill_tensor_tree(None, action_batch, self._shared_data._actions, non_blocking=True)
        if th.cuda.is_initialized() and self._shared_data._device.type == "cuda": # synchronize always unless cuda has never even been used (this way we save a lot of memory)
            th.cuda.synchronize(self._shared_data._device)
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
        ret =  self._shared_data._actions
        return ret


    def wait_data(self) -> Tuple[TensorTree[th.Tensor],
                                th.Tensor,
                                th.Tensor,
                                th.Tensor,
                                TensorTree[th.Tensor],
                                TensorTree[th.Tensor],
                                TensorTree[th.Tensor]]:
        with self._n_envs_stepping_cond:
            didnt_timeout = self._n_envs_stepping_cond.wait_for(lambda: self._n_envs_stepping.value<=0, timeout=self._timeout_s)
        if not didnt_timeout:
            raise RuntimeError(f"SharedEnvData wait timed out waiting for steps")
        return (self._shared_data._consequent_observations,
                self._shared_data._rewards,
                self._shared_data._terminations,
                self._shared_data._truncations,
                self._shared_data._consequent_infos,
                self._shared_data._next_start_observations,
                self._shared_data._next_start_infos)















def example_worker_func(sh : SharedEnvData, sc, worker_id, receiver):
    # print(f"worker starting with {sh} and {steps}")
    import time
    i = 0
    running = True
    device = "cuda"
    while running:
        if i%10000 == 0:
            print(f"worker step {i}")
        # print(f"waiting command...")
        cmd = sc.wait_command()
        # print(f"got command {cmd}")
        i += 1
        if cmd == b"step":
            with th.no_grad():
                act = sh.wait_actions()[worker_id].detach().clone().to(device = device)
            obs = sh.data().observation_space().sample()
            obs = map_tensor_tree(obs, lambda t: t.to(device=device))
            info = {"obs_act_sum" : th.stack([th.sum(obs["obs"]), th.sum(act)])}
            sh.fill_data(   env_idx = worker_id,
                            consequent_observation=obs,
                            action = act,
                            terminated = True,
                            truncated = True,
                            reward = 4,
                            consequent_info = info,
                            next_start_info=info,
                            next_start_observation=obs)
        elif cmd == b"reset":
            obs = sh.data().observation_space().sample()
            act = sh.data().action_space().sample().to(device=device)
            info = {"obs_act_sum" : th.stack([th.sum(obs["obs"]), th.sum(act)])}
            sh.fill_data(   env_idx = worker_id,
                            consequent_observation=obs,
                            action = act,
                            terminated = None,
                            truncated = None,
                            reward = None,
                            consequent_info = info,
                            next_start_info = info,
                            next_start_observation=obs)
        elif cmd == b"close":
            running = False
        elif cmd == b"set_backend_data":
            # print(f"[{worker_id}] setting backend data")
            data : SharedData = receiver.recv()
            sh.set_data_struct(data)
            sh.data().action_space().seed(worker_id)
            sh.data().observation_space().seed(worker_id)
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
    obs_shape = (3,64,64)
    act_space = (8,)
    th_device = th.device("cpu")


    ctx = mp_helper.get_context("forkserver")
    info_example = {"obs_act_sum":th.tensor([0.0,0])}
    info_space = space_from_tree(info_example)
    print(f"info_space = {info_space}")
    sc = SimpleCommander(ctx, n_envs=n_envs, timeout_s=30)
    sh = SharedEnvData(n_envs=n_envs,timeout_s=5, mp_context=ctx)
    receivers, senders = zip(*[ctx.Pipe(duplex=False) for _ in range(n_envs)])
    workers = [ctx.Process(target=example_worker_func, args=(sh,sc,idx, receivers[idx])) for idx in range(n_envs)]
    for worker in workers:
        worker.start()
    data = SharedData(observation_space=gym_spaces.Dict({"obs":ThBox(low = th.ones(size=obs_shape, dtype=th.float32)*-1, high=th.ones(size=obs_shape, dtype=th.float32))}),
                       action_space=ThBox(low = th.ones(size=act_space, dtype=th.float32)*-1, high=th.ones(size=act_space, dtype=th.float32)),
                       info_space=info_space,
                    n_envs=n_envs, device=th_device)
    
    sh.set_data_struct(data)
    sc.set_command("set_backend_data")
    for idx in range(n_envs): senders[idx].send(data)
    print("sent backend data")
    sc.wait_done()
    print("backend data set")
    print("Starting test")
    t0 = time.monotonic()
    errors = 0
    for i in range(steps):
        if i>0 and i%10000 == 0:
            tf = time.monotonic()
            print(f"main step {i} # erorrs={errors} # {tf-t0:.2f}s, {i*n_envs/(tf-t0):.2f} fps, {(tf-t0)/(i*n_envs)}s per step")
        sc.set_command("step")
        sh.mark_waiting_data()
        sh.fill_actions(sh.data().action_space().sample())
        sc.wait_done()

        data = sh.wait_data()
        data = map_tensor_tree(data, lambda t: t.to(device="cuda"))
        # print(f"{th.sum(data[0]['obs'], dim=(1,2,3))} != {data[4]['obs_act_sum'][:,0]} =",th.sum(data[0]["obs"], dim=(1,2,3)) != data[4]["obs_act_sum"][:,0])
        # cobss, rews, terms, truncs, infos, nobs, ninfo = data
        # print(f"obs['obs'] shape = {cobss['obs'].size()}")
        # print(f"obs['obs'][0] sum = {th.sum(cobss['obs'][0])}")
        # print(f"obs['obs'][1] sum = {th.sum(cobss['obs'][1])}")
        # print(f"infos['obs_act_sum'] shape = {infos['obs_act_sum'].size()}")
        # print(f"{th.sum(data[0]['obs'], dim=(1,2,3))} != {data[4]['obs_act_sum']}")
        if not is_all_finite(data):
            print(f"Non-finite values in received data")
            errors += 1
        elif th.any(th.sum(data[0]["obs"], dim=(1,2,3)) != data[4]["obs_act_sum"][:,0]):
            print(f" Obs checksum failed")
            errors += 1
        # print(f"main step {i} end")
    tf = time.monotonic()
    print(f"main(): took {tf-t0}, {steps*n_envs/(tf-t0)} fps, {(tf-t0)/(steps*n_envs)}s per step")
    print(f" Errors: {errors}")
    # print(" # \n".join([str(d) for d in data]))
    sh.mark_waiting_data()
    sc.set_command("reset")
    sh.wait_data()
    print(f"resetted")
    print(" # \n".join([str(d) for d in data]))
    for idx in range(n_envs): senders[idx].close()



    sc.set_command("close")

    worker.join()
    