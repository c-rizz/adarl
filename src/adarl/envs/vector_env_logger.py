from __future__ import annotations
import gymnasium as gym
import numpy as np
import torch as th
from typing import Any, SupportsFloat, Tuple, Dict
import adarl.utils.dbg.ggLog as ggLog
from adarl.utils.tensor_trees import unstack_tensor_tree, filter_tensor_tree
import copy
import adarl.utils.session as session
import time

class VectorEnvLogger(
    gym.vector.VectorEnvWrapper, gym.utils.RecordConstructorArgs
):
    """ Logs metrics from a vector_env """

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        use_wandb : bool = True,
        logs_id : str | None = None
    ):
        gym.Wrapper.__init__(self, env)
        self._current_infos = []
        self._tot_ep_count = 0
        self._use_wandb = use_wandb
        self._logs_batch = {}
        self._logs_batch_size = 0
        self._logs_id = logs_id+"/" if logs_id is not None else ""
        self.__step_count = 0
        self._step_count_last_log = self.__step_count
        self._time_last_log = time.monotonic()
        self._num_envs = env.num_envs


    def step(
        self, action
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Steps through the environment.

        Args:
            action: The environment step action

        Returns:
            The environment step ``(observation, reward, terminated, truncated, info)``

        """
        observation, reward, terminated, truncated, infos = self.env.step(action)
        self.__step_count += 1

        # ggLog.info(f"infos = {infos}")
        # ggLog.info(f"terminated,truncated = {terminated,truncated}")
        # vec_infos = filter_tensor_tree(infos,    keep = lambda t:     (isinstance(t, th.Tensor) and t.dim()>0 and t.size()[0] == self._num_envs))
        # nonvec_infos = filter_tensor_tree(infos, keep = lambda t: not (isinstance(t, th.Tensor) and t.dim()>0 and t.size()[0] == self._num_envs))
        # ggLog.info(f"vec_infos = {vec_infos}")
        # ggLog.info(f"nonvec_infos = {nonvec_infos}")
        final_infos = infos["final_info"]
        final_infos = {k:v for k,v in final_infos.items() if k != "final_info"} # make a shallow copy without the final_info cycle
        # infos.pop("final_infos")
        # info_list = unstack_tensor_tree(infos)
        final_info_list = unstack_tensor_tree(final_infos)
        if self._use_wandb:
            from adarl.utils.wandb_wrapper import wandb_log
            for i in range(self._num_envs):
                if terminated[i] or truncated[i]: # we only log the info of the last step
                    self._tot_ep_count += 1
                    info = final_info_list[i]
                    logs = {}
                    for k,v in info.items():
                        k = "VecEnvLogger/lastinfo."+k
                        if isinstance(v,dict):
                            # ggLog.info(f"flattening {k}:{v}")
                            for k1,v1 in v.items():
                                logs[k+"."+k1] = v1
                        else:
                            if type(v) is bool:
                                v = int(v)
                            logs[k] = v
                    logs["VecEnvLogger/vec_ep_count"] = self._tot_ep_count
                    logs = copy.deepcopy(logs) # avoid issues with references (yes, it does happen)
                    for k in logs.keys():
                        if k not in self._logs_batch:
                            self._logs_batch[k] = []
                        self._logs_batch[k].append(logs[k])
                    self._logs_batch_size +=1
            if self._logs_batch_size >= self._num_envs:
                new_elems = {}
                for k,v in self._logs_batch.items():
                    if len(v)>0 and isinstance(v[0],(int, float, bool, np.integer, np.floating, th.Tensor)):
                        self._logs_batch[k] = sum(v)/len(v)
                        if isinstance(v[0],(int, float, bool, np.integer, np.floating)) or v[0].numel()==1:  # only if v has just on element
                            new_elems[k.replace("VecEnvLogger/","VecEnvLogger/max.")] = max(v)
                            new_elems[k.replace("VecEnvLogger/","VecEnvLogger/min.")] = min(v)
                self._logs_batch.update(new_elems)
                wdblog = {k: v.cpu().item() if isinstance(v,th.Tensor) and v.numel()==1 else v for k,v in self._logs_batch.items()}
                wdblog = {self._logs_id+k: v.cpu().item() if isinstance(v,th.Tensor) and v.numel()==1 else v for k,v in self._logs_batch.items()}
                wandb_log(wdblog)
                ggLog.info(f"{self._logs_id}VecEnvLogger: tot_ep_count={self._tot_ep_count} veceps={int(self._tot_ep_count/self._num_envs)} succ={self._logs_batch.get('VecEnvLogger/success',0):.2f}"+
                           f" r= \033[1m{self._logs_batch.get('VecEnvLogger/lastinfo.ep_reward',float('nan')):08.8g}\033[0m "+
                           f" min_r={self._logs_batch.get('VecEnvLogger/min.lastinfo.ep_reward',float('nan')):08.8g}"
                           f" max_r={self._logs_batch.get('VecEnvLogger/max.lastinfo.ep_reward',float('nan')):08.8g}"
                           f" fps={self._num_envs*(self.__step_count-self._step_count_last_log)/(time.monotonic() - self._time_last_log):.2f}")
                self._logs_batch = {}
                self._logs_batch_size = 0
                self._step_count_last_log = self.__step_count
                self._time_last_log = time.monotonic()
        return observation, reward, terminated, truncated, infos