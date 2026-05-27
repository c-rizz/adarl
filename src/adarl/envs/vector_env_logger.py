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
from adarl.utils.utils import masked_assign
from adarl.utils.spaces import get_1d_space_size
import pprint
from adarl.utils.async_cuda2cpu_queue import log_async

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
        env_th_device : th.device | str = th.device("cuda")
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
        self._env_th_device = env_th_device if isinstance(env_th_device, th.device) else th.device(env_th_device)
        if hasattr(env.unwrapped, "single_reward_space"):
            reward_space = env.unwrapped.single_reward_space
        else:
            reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        rewards_num = get_1d_space_size(reward_space)

        self._ep_rewards = th.zeros(size=(self._num_envs,rewards_num)).to(device=self._env_th_device, non_blocking=self._env_th_device.type=="cuda")
        self._ep_durations = th.zeros(size=(self._num_envs,), dtype=th.long).to(device=self._env_th_device, non_blocking=self._env_th_device.type=="cuda")
        self._completed_ep_rewards_sum_sl : th.Tensor =   th.as_tensor(0.0).to(device=self._env_th_device, non_blocking=self._env_th_device.type=="cuda")
        self._completed_ep_rewards_min_sl : th.Tensor =   th.as_tensor(float("+inf")).to(device=self._env_th_device, non_blocking=self._env_th_device.type=="cuda")
        self._completed_ep_rewards_max_sl : th.Tensor =   th.as_tensor(float("-inf")).to(device=self._env_th_device, non_blocking=self._env_th_device.type=="cuda")
        self._completed_ep_durations_sum_sl : th.Tensor = th.as_tensor(0.0).to(device=self._env_th_device, non_blocking=self._env_th_device.type=="cuda")
        self._completed_ep_durations_min_sl : th.Tensor = th.as_tensor(float("+inf")).to(device=self._env_th_device, non_blocking=self._env_th_device.type=="cuda")
        self._completed_ep_durations_max_sl : th.Tensor = th.as_tensor(float("-inf")).to(device=self._env_th_device, non_blocking=self._env_th_device.type=="cuda")
        self._completed_ep_count_sl = 0
        self._tot_completed_ep_count = 0
        self._overhead_count = 0
        self._overhead_sum = 0
        self._overhead_max = float("-inf")
        self._overhead_min = float("+inf")
        self._completed_final_infos_since_log : dict[str,th.Tensor] ={}


    def reset(self, *, seed: int | session.List[int] | None = None, options: Dict | None = None):
        reset_stats = options.get("vec_env_logger_reset_stats", True) if options is not None else True
        if reset_stats:
            self._reset_stats()
        return super().reset(seed=seed, options=options)

    def _reset_stats(self):
        # self._ep_rewards.fill_(0.0)
        # self._ep_durations.fill_(0)
        self._completed_ep_rewards_sum_sl.fill_(0.0)
        self._completed_ep_rewards_min_sl.fill_(float("+inf"))
        self._completed_ep_rewards_max_sl.fill_(float("-inf"))
        self._completed_ep_durations_sum_sl.fill_(0.0)
        self._completed_ep_durations_min_sl.fill_(float("+inf"))
        self._completed_ep_durations_max_sl.fill_(float("-inf"))
        self._completed_ep_count_sl = 0
        self._overhead_count = 0
        self._overhead_sum = 0
        self._overhead_max = float("-inf")
        self._overhead_min = float("+inf")
        self._time_last_log = time.monotonic()
        self._step_count_last_log = self.__vstep_count
        for k in self._completed_final_infos_since_log:            
            t = self._completed_final_infos_since_log[k]
            self._completed_final_infos_since_log[k] = th.full_like(t, fill_value = float("nan"))

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
        self._ep_rewards += reward.view(-1, self._ep_rewards.shape[1])
        self._ep_durations += 1
        completed_eps = th.logical_or(th_terminateds, th_truncateds)
        newly_completed_eps_count = completed_eps.count_nonzero()
        if newly_completed_eps_count>0:
            # ggLog.info(f"self._ep_rewards = {self._ep_rewards}")
            tot_rewards = self._ep_rewards.sum(dim=1)
            completed_tot_rewards = th.masked.masked_tensor(tot_rewards, completed_eps)
            completed_durs = th.masked.masked_tensor(self._ep_durations, completed_eps)
            # ggLog.info(f"{self._logs_id}VecEnvLogger: completed_durs = {self._ep_durations[completed_eps]}")
            self._completed_ep_rewards_sum_sl += completed_tot_rewards.sum(dim=0).to_tensor(0)
            self._completed_ep_rewards_min_sl = th.minimum(th.amin(completed_tot_rewards, dim=0).to_tensor(0), self._completed_ep_rewards_min_sl)
            self._completed_ep_rewards_max_sl = th.maximum(th.amax(completed_tot_rewards, dim=0).to_tensor(0), self._completed_ep_rewards_max_sl)
            self._completed_ep_durations_sum_sl += th.sum(completed_durs).to_tensor(0)
            self._completed_ep_durations_min_sl = th.minimum(th.amin(completed_durs).to_tensor(0), self._completed_ep_durations_min_sl)
            self._completed_ep_durations_max_sl = th.maximum(th.amax(completed_durs).to_tensor(0), self._completed_ep_durations_max_sl)
            masked_assign(self._ep_rewards, completed_eps, 0.0)
            masked_assign(self._ep_durations, completed_eps, 0)
            
            if self._log_infos:

                final_infos = infos["final_info"]
                final_infos = {k:v for k,v in final_infos.items() if k != "final_info"} # make a shallow copy without the final_info cycle
                final_infos = flatten_tensor_tree(final_infos)
                final_infos = {"lastinfo."+(".".join(k)):v for k,v in final_infos.items()} # convert keys to strings
                final_infos = {k:int(v) if isinstance(v, bool) else v for k,v in final_infos.items()}
                final_infos = {k:th.as_tensor(v) if isinstance(v, (int,float,bool,np.ndarray,np.number)) else v for k,v in final_infos.items()}
                final_infos = {k:v for k,v in final_infos.items() if isinstance(v,th.Tensor)}
                final_infos = {k:v.reshape(-1) for k,v in final_infos.items()}
                final_infos = {k:v for k,v in final_infos.items() if v.size()==(self._num_envs,)}
                final_infos = {k:v.to(dtype=th.float32) for k,v in final_infos.items()}
                for k in final_infos:
                    if k not in self._completed_final_infos_since_log:
                        t = final_infos[k]
                        max_size = t.size()[0]*2
                        self._completed_final_infos_since_log[k] = th.full(fill_value = float("nan"),
                                                                            size = (max_size,)+t.size()[1:],
                                                                            device=t.device)

                #Would be nice to do the following just with masks, avoiding
                completed_final_infos = {k:v[completed_eps] for k,v in final_infos.items()}
                for k in final_infos:
                    # ggLog.info(f"[{k}][{self._completed_eps_since_log}:{self._completed_eps_since_log+completed_eps_count}]={completed_final_infos[k].size()}")
                    self._completed_final_infos_since_log[k][self._completed_ep_count_sl:self._completed_ep_count_sl+newly_completed_eps_count] = completed_final_infos[k]
            
            self._completed_ep_count_sl += newly_completed_eps_count
            self._tot_completed_ep_count += newly_completed_eps_count
            # ggLog.info(f"{self._logs_id}VecEnvLogger: {newly_completed_eps_count} episodes newly completed, total completed={self._tot_completed_ep_count}, since last log={self._completed_ep_count_sl}")
            # ggLog.info(f" completed ep durations = {completed_durs}, sum={self._completed_ep_durations_sum_sl}, min={self._completed_ep_durations_min_sl}, max={self._completed_ep_durations_max_sl}")
            # ggLog.info(f" terminated = {th_terminateds}, truncated={th_truncateds}, completed_eps={completed_eps}")
            # ggLog.info(f" ep durations = {self._ep_durations}, ep rewards = {self._ep_rewards}")
            if self._completed_ep_count_sl >= self._num_envs:
                ravg = self._completed_ep_rewards_sum_sl/self._completed_ep_count_sl
                davg = self._completed_ep_durations_sum_sl/self._completed_ep_count_sl            
                log_async(f"{self._logs_id}VecEnvLogger:"
                           " ep={tot_completed_ep_count}"
                           " reward avg={ravg},"
                           " min={completed_ep_rewards_min_sl},"
                           " max={completed_ep_rewards_max_sl},"
                           " length={davg}[{completed_ep_durations_min_sl},{completed_ep_durations_max_sl}]",
                           tensors=dict(ravg=ravg,
                                        tot_completed_ep_count=self._tot_completed_ep_count,
                                        completed_ep_rewards_min_sl=self._completed_ep_rewards_min_sl,
                                        completed_ep_rewards_max_sl=self._completed_ep_rewards_max_sl,
                                        davg=davg,
                                        completed_ep_durations_min_sl=self._completed_ep_durations_min_sl,
                                        completed_ep_durations_max_sl=self._completed_ep_durations_max_sl))
                wdblog = {'VecEnvLogger/'+self._logs_id+"/reward_avg": ravg,
                          'VecEnvLogger/'+self._logs_id+"/reward_min": self._completed_ep_rewards_min_sl,
                          'VecEnvLogger/'+self._logs_id+"/reward_max": self._completed_ep_rewards_max_sl,
                          'VecEnvLogger/'+self._logs_id+"/length_avg": davg,
                          'VecEnvLogger/'+self._logs_id+"/length_min": self._completed_ep_durations_min_sl,
                          'VecEnvLogger/'+self._logs_id+"/length_max": self._completed_ep_durations_max_sl}
                if self._log_infos:
                    logs = {}
                    logged_infos = {k:v[:self._completed_ep_count_sl] for k,v in self._completed_final_infos_since_log.items()}
                    # ggLog.info(f"{self._logs_id}VecEnvLogger: logged_infos keys: {list(logged_infos.keys())}")
                    # ggLog.info(f"{self._logs_id}VecEnvLogger: logged_infos[ep_Reward]: {logged_infos['lastinfo.ep_reward']}")
                    logs.update({"VecEnvLogger/avg."+k:v.mean() for k,v in logged_infos.items()})
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
                    avgrew = logs.get('VecEnvLogger/avg.lastinfo.ep_reward',None)
                    minrew = logs.get('VecEnvLogger/min.lastinfo.ep_reward',None)
                    maxrew = logs.get('VecEnvLogger/max.lastinfo.ep_reward',None)
                    medrew = logs.get('VecEnvLogger/med.lastinfo.ep_reward',None)

                    fps=self._num_envs*(self.__vstep_count-self._step_count_last_log)/(time.monotonic() - self._time_last_log)
                    log_async( f"{self._logs_id}VecEnvLogger:"+
                                f" tot_ep_count={self._tot_completed_ep_count}"+
                                f" veceps={int(self._tot_completed_ep_count/self._num_envs)}"+
                                f" succ={logs.get('VecEnvLogger/success',0):.2f}"+
                                f" r=\033[1m"+  ("{avgrew:08.8g}" if avgrew is not None else "???") +"\033[0m "+
                                f" min_r="+     ("{minrew:08.8g}" if minrew is not None else "???")+
                                f" max_r="+     ("{maxrew:08.8g}" if maxrew is not None else "???")+
                                f" med_r="+     ("{medrew:08.8g}" if medrew is not None else "???")+
                                f" fps={fps:.2f}",
                                tensors=dict(avgrew=avgrew,
                                             minrew=minrew,
                                             maxrew=maxrew,
                                             medrew=medrew))
                    if self._use_wandb:
                        wdblog.update({f"{k.replace('VecEnvLogger/','VecEnvLogger/'+self._logs_id)}": v.cpu().item() if isinstance(v,th.Tensor) and v.numel()==1 else v for k,v in logs.items()})
                    # ggLog.info(f"Logger overhead: {self._overhead_sum/self._overhead_count:.9f}[{self._overhead_min},{self._overhead_max}]")                    
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
                if self._use_wandb:
                    from adarl.utils.wandb_wrapper import wandb_log
                    wandb_log(wdblog)
                self._reset_stats()

        tf = time.monotonic()
        overhead = (tf-t1)/(t1-t0)
        self._overhead_count += 1
        self._overhead_sum += overhead
        self._overhead_max = max(overhead, self._overhead_max)
        self._overhead_min = min(overhead, self._overhead_min)

        return observation, reward, terminated, truncated, infos