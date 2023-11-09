import time
import warnings
from typing import Optional, Tuple

import numpy as np
import lr_gym.utils.dbg.ggLog as ggLog

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper


class VecEnvLogger(VecEnvWrapper):
    """
    A vectorized monitor wrapper for *vectorized* Gym environments,
    it is used to record the episode reward, length, time and other data.

    Some environments like `openai/procgen <https://github.com/openai/procgen>`_
    or `gym3 <https://github.com/openai/gym3>`_ directly initialize the
    vectorized environments, without giving us a chance to use the ``Monitor``
    wrapper. So this class simply does the job of the ``Monitor`` wrapper on
    a vectorized level.

    :param venv: The vectorized environment
    :param filename: the location to save a log file, can be None for no log
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    def __init__(
        self,
        venv: VecEnv,
        use_wandb : bool = True
    ):

        VecEnvWrapper.__init__(self, venv)
        self._current_infos = []
        self._tot_ep_count = 0
        self._use_wandb = use_wandb
        self._logs_batch = {}
        self._logs_batch_size = 0

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        if self._use_wandb:
            from lr_gym.utils.wandb_wrapper import wandb_log
            for i in range(len(dones)):
                if dones[i]:
                    self._tot_ep_count += 1
                    info = infos[i]
                    logs = {"VecEnvLogger/"+str(k) : v if type(v) is not bool else int(v) for k,v in info.items()}
                    # ggLog.info(f"logging {logs}")
                    logs["VecEnvLogger/vec_ep_count"] = self._tot_ep_count
                    logs["vec_ep_count"] = self._tot_ep_count # for compatibility, to be removed
                    for k in logs.keys():
                        if k not in self._logs_batch:
                            self._logs_batch[k] = []
                        self._logs_batch[k].append(logs[k])
                    self._logs_batch_size +=1
            if self._logs_batch_size >= self.num_envs:
                for k,v in self._logs_batch.items():
                    # ggLog.info(f"k = {k}")
                    if len(v)>0 and isinstance(v[0],(int, float, bool, np.integer, np.floating)):
                        self._logs_batch[k] = sum(v)/len(v)
                wandb_log(lambda: self._logs_batch)
                ggLog.info(f"VecEnvLogger: tot_ep_count={self._tot_ep_count} success={self._logs_batch.get('VecEnvLogger/success',0):.2f} reward={self._logs_batch.get('VecEnvLogger/ep_reward',0):9g}")
                self._logs_batch = {}
                self._logs_batch_size = 0
        return obs, rewards, dones, infos

    def close(self) -> None:
        return self.venv.close()
