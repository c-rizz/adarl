from __future__ import annotations
import gymnasium as gym
import numpy as np
import torch as th
from typing import Any, SupportsFloat, Tuple, Dict
import adarl.utils.dbg.ggLog as ggLog
from adarl.utils.tensor_trees import unstack_tensor_tree, filter_tensor_tree, TensorTree, TensorMapping, flatten_tensor_tree, map_tensor_tree
import copy
import adarl.utils.session as session
import time
from adarl.utils.utils import masked_assign_sc
import pprint
class VectorEnvLogger(
    gym.vector.VectorEnvWrapper, gym.utils.RecordConstructorArgs
):
    """ Logs metrics from a vector_env """

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        use_wandb : bool = True,
        logs_id : str | None = None,
        log_infos : bool =  True,
        env_th_device : th.device = th.device("cuda")
    ):
        gym.Wrapper.__init__(self, env)
        self._current_infos = []
        self._use_wandb = use_wandb
        self._logs_batch = {}
        self._logs_batch_size = 0
        self._logs_id = logs_id+"/" if (logs_id is not None and logs_id!="") else ""
        self.__vstep_count = 0
        self._step_count_last_log = self.__vstep_count
        self._time_last_log = time.monotonic()
        self._log_infos = log_infos
        self._num_envs = env.unwrapped.num_envs
        self._env_th_device = env_th_device

        self._ep_rewards = th.zeros(size=(self._num_envs,), device=self._env_th_device)
        self._ep_durations = th.zeros(size=(self._num_envs,), device=self._env_th_device, dtype=th.long)
        self._completed_ep_rewards_sum_sl = th.as_tensor(0.0, device=self._env_th_device)
        self._completed_ep_rewards_min_sl = th.as_tensor(float("+inf"), device=self._env_th_device)
        self._completed_ep_rewards_max_sl = th.as_tensor(float("-inf"), device=self._env_th_device)
        self._completed_ep_durations_sum_sl = th.as_tensor(0.0, device=self._env_th_device)
        self._completed_ep_durations_min_sl = th.as_tensor(float("+inf"), device=self._env_th_device)
        self._completed_ep_durations_max_sl = th.as_tensor(float("-inf"), device=self._env_th_device)
        self._completed_ep_count_sl = 0
        self._tot_completed_ep_count = 0
        self._overhead_count = 0
        self._overhead_sum = 0
        self._overhead_max = float("-inf")
        self._overhead_min = float("+inf")
        self._completed_final_infos_since_log : dict[str,th.Tensor] ={}
        self._completed_eps_since_log = 0


    def reset(self, *, seed: int | session.List[int] | None = None, options: Dict | None = None):
        return super().reset(seed=seed, options=options)

    def step(
        self, action
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Steps through the environment.

        Args:
            action: The environment step action

        Returns:
            The environment step ``(observation, reward, terminated, truncated, info)``

        """
        t0 = time.monotonic()
        terminated : th.Tensor
        truncated : th.Tensor
        reward : th.Tensor
        observation : TensorMapping[th.Tensor]
        infos : TensorMapping[th.Tensor]
        observation, reward, terminated, truncated, infos = self.env.step(action)
        t1 = time.monotonic()
        self.__vstep_count += 1

        th_terminateds, th_truncateds = th.as_tensor(terminated), th.as_tensor(truncated)
        self._ep_rewards += reward
        self._ep_durations += 1
        completed_eps = th.logical_or(th_terminateds, th_truncateds)
        completed_eps_count = completed_eps.count_nonzero()
        if completed_eps_count>0:
            completed_rews = self._ep_rewards[completed_eps]
            completed_durs = self._ep_durations[completed_eps]
            self._completed_ep_rewards_sum_sl += th.sum(completed_rews)
            self._completed_ep_rewards_min_sl = th.min(th.min(completed_rews), self._completed_ep_rewards_min_sl)
            self._completed_ep_rewards_max_sl = th.max(th.max(completed_rews), self._completed_ep_rewards_max_sl)
            self._completed_ep_durations_sum_sl += th.sum(completed_durs)
            self._completed_ep_durations_min_sl = th.min(th.min(completed_durs), self._completed_ep_durations_min_sl)
            self._completed_ep_durations_max_sl = th.max(th.max(completed_durs), self._completed_ep_durations_max_sl)
            self._completed_ep_count_sl += completed_eps_count
            self._tot_completed_ep_count += completed_eps_count
            if self._completed_ep_count_sl >= self._num_envs:
                ravg = self._completed_ep_rewards_sum_sl/self._completed_ep_count_sl
                davg = self._completed_ep_durations_sum_sl/self._completed_ep_count_sl
                ggLog.info(f"VecEnvLogger: ep={self._tot_completed_ep_count} reward avg={ravg}, min={self._completed_ep_rewards_min_sl}, max={self._completed_ep_rewards_max_sl}, length={davg}[{self._completed_ep_durations_min_sl},{self._completed_ep_durations_max_sl}]")
                self._completed_ep_rewards_sum_sl.fill_(0.0)
                self._completed_ep_rewards_min_sl.fill_(float("+inf"))
                self._completed_ep_rewards_max_sl.fill_(float("-inf"))
                self._completed_ep_durations_sum_sl.fill_(0.0)
                self._completed_ep_durations_min_sl.fill_(float("+inf"))
                self._completed_ep_durations_max_sl.fill_(float("-inf"))
                self._completed_ep_count_sl = 0
            self._ep_rewards[completed_eps] = 0.0
            self._ep_durations[completed_eps] = 0
            

            if self._log_infos:
                # ggLog.info(f"infos = {infos}")
                # ggLog.info(f"terminated,truncated = {terminated,truncated}")
                # vec_infos = filter_tensor_tree(infos,    keep = lambda t:     (isinstance(t, th.Tensor) and t.dim()>0 and t.size()[0] == self._num_envs))
                # nonvec_infos = filter_tensor_tree(infos, keep = lambda t: not (isinstance(t, th.Tensor) and t.dim()>0 and t.size()[0] == self._num_envs))
                # ggLog.info(f"vec_infos = {vec_infos}")
                # ggLog.info(f"nonvec_infos = {nonvec_infos}")
                # infos.pop("final_infos")
                # info_list = unstack_tensor_tree(infos)

                # if th.any(th.logical_or(th_terminateds, th_truncateds)):
                #     final_infos = infos["final_info"]
                #     final_infos = {k:v for k,v in final_infos.items() if k != "final_info"} # make a shallow copy without the final_info cycle
                #     final_info_list = unstack_tensor_tree(final_infos)
                #     for i in range(self._num_envs):
                #         if terminated[i] or truncated[i]: # we only log the info of the last step
                #             info = final_info_list[i]
                #             logs = {}
                #             for k,v in info.items():
                #                 k = "VecEnvLogger/lastinfo."+k
                #                 if isinstance(v,dict):
                #                     # ggLog.info(f"flattening {k}:{v}")
                #                     for k1,v1 in v.items():
                #                         logs[k+"."+k1] = v1
                #                 else:
                #                     if type(v) is bool:
                #                         v = int(v)
                #                     logs[k] = v
                #             logs["VecEnvLogger/vec_ep_count"] = self._tot_completed_ep_count
                #             logs = copy.deepcopy(logs) # avoid issues with references (yes, it does happen)
                #             for k in logs.keys():
                #                 if k not in self._logs_batch:
                #                     self._logs_batch[k] = []
                #                 self._logs_batch[k].append(logs[k])
                #             self._logs_batch_size +=1
                # if self._logs_batch_size >= self._num_envs:
                #     # ggLog.info(f"logging veclogger, wandb={self._use_wandb}")
                #     wall_single_fps = (self.__vstep_count - self._step_count_last_log)/(time.monotonic()-self._time_last_log)
                #     new_elems = {}
                #     for k,v in self._logs_batch.items():
                #         if len(v)>0 and isinstance(v[0],(int, float, bool, np.integer, np.floating, th.Tensor)):
                #             self._logs_batch[k] = sum(v)/len(v)
                #             if isinstance(v[0],(int, float, bool, np.integer, np.floating)) or v[0].numel()==1:  # only if v has just on element
                #                 new_elems[k.replace("VecEnvLogger/","VecEnvLogger/max.")] = max(v)
                #                 new_elems[k.replace("VecEnvLogger/","VecEnvLogger/min.")] = min(v)
                #     self._logs_batch.update(new_elems)
                #     self._logs_batch["VecEnvLogger/wall_fps_vec"] = wall_single_fps*self.num_envs
                #     self._logs_batch["VecEnvLogger/wall_fps_single"] = wall_single_fps
                #     self._logs_batch["VecEnvLogger/vec_ep_count"] = self._tot_completed_ep_count
                #     ggLog.info(f"{self._logs_id}VecEnvLogger: tot_ep_count={self._tot_completed_ep_count} veceps={int(self._tot_completed_ep_count/self._num_envs)} succ={self._logs_batch.get('VecEnvLogger/success',0):.2f}"+
                #             f" r= \033[1m{self._logs_batch.get('VecEnvLogger/lastinfo.ep_reward',float('nan')):08.8g}\033[0m "+
                #             f" min_r={self._logs_batch.get('VecEnvLogger/min.lastinfo.ep_reward',float('nan')):08.8g}"
                #             f" max_r={self._logs_batch.get('VecEnvLogger/max.lastinfo.ep_reward',float('nan')):08.8g}"
                #             f" fps={self._num_envs*(self.__vstep_count-self._step_count_last_log)/(time.monotonic() - self._time_last_log):.2f}")
                #     if self._use_wandb:
                #         from adarl.utils.wandb_wrapper import wandb_log
                #         # ggLog.info(f"vecenvlogger logging: {list(self._logs_batch.keys())}")
                #         wdblog = {f"{self._logs_id}{k}": v.cpu().item() if isinstance(v,th.Tensor) and v.numel()==1 else v for k,v in self._logs_batch.items()}
                #         wandb_log(wdblog)
                #     ggLog.info(f"Logger overhead: {self._overhead_sum/self._overhead_count:.9f}[{self._overhead_min},{self._overhead_max}]")                    
                #     self._logs_batch = {}
                #     self._logs_batch_size = 0
                #     self._step_count_last_log = self.__vstep_count
                #     self._time_last_log = time.monotonic()
                #     self._overhead_count = 0
                #     self._overhead_sum = 0
                #     self._overhead_max = float("-inf")
                #     self._overhead_min = float("+inf")

                if th.any(completed_eps):
                    final_infos = infos["final_info"]
                    final_infos = {k:v for k,v in final_infos.items() if k != "final_info"} # make a shallow copy without the final_info cycle
                    final_infos = flatten_tensor_tree(final_infos)
                    final_infos = {"lastinfo."+(".".join(k)):v for k,v in final_infos.items()} # convert keys to strings
                    final_infos = map_tensor_tree(final_infos, lambda l: int(l) if isinstance(l, bool) else l)
                    final_infos = map_tensor_tree(final_infos,
                                                            lambda l: th.as_tensor(l) if isinstance(l, (int, float, bool, np.ndarray, np.number)) else l)
                    final_infos = {k:v for k,v in final_infos.items() if isinstance(v,th.Tensor)}
                    final_infos = {k:v.reshape(-1) for k,v in final_infos.items()}
                    final_infos = {k:v for k,v in final_infos.items() if v.view(-1).size()==(self._num_envs,)}
                    final_infos = {k:v.to(dtype=th.float32) for k,v in final_infos.items()}
                    for k in final_infos:
                        if k not in self._completed_final_infos_since_log:
                            t = final_infos[k]
                            max_size = t.size()[0]*2
                            self._completed_final_infos_since_log[k] = th.full(fill_value = float("nan"),
                                                                               size = (max_size,)+t.size()[1:],
                                                                               device=t.device)

                    #Would be nice to do the following just with masks, avoiding
                    completed_eps_count = th.count_nonzero(completed_eps)
                    completed_final_infos = {k:v[completed_eps] for k,v in final_infos.items()}
                    for k in final_infos:
                        # ggLog.info(f"[{k}][{self._completed_eps_since_log}:{self._completed_eps_since_log+completed_eps_count}]={completed_final_infos[k].size()}")
                        self._completed_final_infos_since_log[k][self._completed_eps_since_log:self._completed_eps_since_log+completed_eps_count] = completed_final_infos[k]
                    self._completed_eps_since_log += completed_eps_count
                if self._completed_eps_since_log >= self._num_envs:
                    logs = {}
                    logged_infos = {k:v[:self._completed_eps_since_log] for k,v in self._completed_final_infos_since_log.items()}
                    # ggLog.info(f"VecEnvLogger: _completed_final_infos_since_log = {pprint.pformat(logged_infos)}")
                    avgs = {k:v.mean() for k,v in logged_infos.items()}

                    logs.update({"VecEnvLogger/avg."+k:v.mean() for k,v in avgs.items()})
                    logs.update({"VecEnvLogger/"+k:v.mean() for k,v in avgs.items()})
                    logs.update({"VecEnvLogger/min."+k:v.min()  for k,v in logged_infos.items()})
                    logs.update({"VecEnvLogger/max."+k:v.max()  for k,v in logged_infos.items()})
                    logs.update({"VecEnvLogger/med."+k:v.median()  for k,v in logged_infos.items()})
                    logs.update({"VecEnvLogger/q95."+k:v.quantile(0.95)  for k,v in logged_infos.items()})
                    logs.update({"VecEnvLogger/q05."+k:v.quantile(0.05)  for k,v in logged_infos.items()})
                    wall_single_fps = (self.__vstep_count - self._step_count_last_log)/(time.monotonic()-self._time_last_log)
                    logs["VecEnvLogger/wall_fps_vec"] = wall_single_fps*self._num_envs
                    logs["VecEnvLogger/wall_fps_single"] = wall_single_fps
                    logs["VecEnvLogger/vec_ep_count"] = self._tot_completed_ep_count
                    # ggLog.info(f"{logs}")
                    ggLog.info(f"{self._logs_id}VecEnvLogger: tot_ep_count={self._tot_completed_ep_count} veceps={int(self._tot_completed_ep_count/self._num_envs)} succ={logs.get('VecEnvLogger/success',0):.2f}"+
                            f" r= \033[1m{logs.get('VecEnvLogger/avg.lastinfo.ep_reward',float('nan')):08.8g}\033[0m "+
                            f" min_r={logs.get('VecEnvLogger/min.lastinfo.ep_reward',float('nan')):08.8g}"
                            f" max_r={logs.get('VecEnvLogger/max.lastinfo.ep_reward',float('nan')):08.8g}"
                            f" med_r={logs.get('VecEnvLogger/med.lastinfo.ep_reward',float('nan')):08.8g}"
                            f" fps={self._num_envs*(self.__vstep_count-self._step_count_last_log)/(time.monotonic() - self._time_last_log):.2f}")
                    if self._use_wandb:
                        from adarl.utils.wandb_wrapper import wandb_log
                        # ggLog.info(f"vecenvlogger logging: {list(logs.keys())}")
                        wdblog = {f"{self._logs_id}{k}": v.cpu().item() if isinstance(v,th.Tensor) and v.numel()==1 else v for k,v in logs.items()}
                        wandb_log(wdblog)
                    # ggLog.info(f"Logger overhead: {self._overhead_sum/self._overhead_count:.9f}[{self._overhead_min},{self._overhead_max}]")                    
                    self._step_count_last_log = self.__vstep_count
                    self._time_last_log = time.monotonic()
                    self._overhead_count = 0
                    self._overhead_sum = 0
                    self._overhead_max = float("-inf")
                    self._overhead_min = float("+inf")
                    self._completed_eps_since_log = 0
                    for k in self._completed_final_infos_since_log:
                        self._completed_final_infos_since_log[k].fill_(float("nan")) # Fill with nans, so that we see if something goes wrong
                    # final_info_list = unstack_tensor_tree(final_infos)
                    # for i in range(self._num_envs):
                    #     if terminated[i] or truncated[i]: # we only log the info of the last step
                    #         info = final_info_list[i]
                    #         logs = {}
                    #         logs = flatten_tensor_tree(info)
                    #         logs = {"VecEnvLogger/lastinfo."+(".".join(k)):v for k,v in logs.items()} # convert keys to strings
                    #         logs = map_tensor_tree(logs, lambda l: int(l) if isinstance(l, bool) else l)
                    #         logs = copy.deepcopy(logs) # avoid issues with references (yes, it does happen)
                    #         for k in logs.keys():
                    #             if k not in self._logs_batch:
                    #                 self._logs_batch[k] = []
                    #             self._logs_batch[k].append(logs[k])
                    #         self._logs_batch_size +=1
        tf = time.monotonic()
        overhead = (tf-t1)/(t1-t0)
        self._overhead_count += 1
        self._overhead_sum += overhead
        self._overhead_max = max(overhead, self._overhead_max)
        self._overhead_min = min(overhead, self._overhead_min)

        return observation, reward, terminated, truncated, infos