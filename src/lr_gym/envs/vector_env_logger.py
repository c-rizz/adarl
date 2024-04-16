import gymnasium as gym
from gymnasium.core import ActType, ObsType, RenderFrame, WrapperObsType
import numpy as np
import torch as th
from typing import Any, SupportsFloat
import lr_gym.utils.dbg.ggLog as ggLog
from lr_gym.utils.tensor_trees import unstack_tensor_tree

class VectorEnvLogger(
    gym.Wrapper[ObsType, ActType, ObsType, ActType], gym.utils.RecordConstructorArgs
):
    """ Logs metrics from a vector_env """

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        use_wandb : bool = True
    ):
        """Initializes the :class:`TimeLimit` wrapper with an environment and the number of steps after which truncation will occur.

        Args:
            env: The environment to apply the wrapper
            max_episode_steps: An optional max episode steps (if ``None``, ``env.spec.max_episode_steps`` is used)
        """
        gym.Wrapper.__init__(self, env)
        self._current_infos = []
        self._tot_ep_count = 0
        self._use_wandb = use_wandb
        self._logs_batch = {}
        self._logs_batch_size = 0


    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment.

        Args:
            action: The environment step action

        Returns:
            The environment step ``(observation, reward, terminated, truncated, info)``

        """
        observation, reward, terminated, truncated, infos = self.env.step(action)

        info_list = unstack_tensor_tree(infos)
        if self._use_wandb:
            from lr_gym.utils.wandb_wrapper import wandb_log
            for i in range(len(terminated)):
                if terminated[i] or truncated[i]:
                    self._tot_ep_count += 1
                    info = info_list[i]
                    logs = {}
                    for k,v in info.items():
                        k = "VecEnvLogger/lastinfo/"+k
                        if isinstance(v,dict):
                            # ggLog.info(f"flattening {k}:{v}")
                            for k1,v1 in v.items():
                                logs[k+"."+k1] = v1
                        else:
                            if type(v) is bool:
                                v = int(v)
                            logs[k] = v
                    logs["VecEnvLogger/vec_ep_count"] = self._tot_ep_count
                    for k in logs.keys():
                        if k not in self._logs_batch:
                            self._logs_batch[k] = []
                        self._logs_batch[k].append(logs[k])
                    self._logs_batch_size +=1
            if self._logs_batch_size >= self.num_envs:
                new_elems = {}
                for k,v in self._logs_batch.items():
                    # ggLog.info(f"k = {k}")
                    if len(v)>0 and isinstance(v[0],(int, float, bool, np.integer, np.floating, th.Tensor)):
                        self._logs_batch[k] = sum(v)/len(v)
                        if isinstance(v[0],(int, float, bool, np.integer, np.floating)) or v[0].numel()==1:  # only if v has just on element
                            new_elems[k.replace("VecEnvLogger/","VecEnvLogger/max.")] = max(v)
                            new_elems[k.replace("VecEnvLogger/","VecEnvLogger/min.")] = min(v)
                self._logs_batch.update(new_elems)
                wdblog = {k: v.cpu().item() if isinstance(v,th.Tensor) and v.numel()==1 else v for k,v in self._logs_batch.items()}
                # ggLog.info(f"wdblog = {wdblog}")
                wandb_log(wdblog)
                ggLog.info(f"VecEnvLogger: tot_ep_count={self._tot_ep_count} veceps={int(self._tot_ep_count/self.num_envs)} succ={self._logs_batch.get('VecEnvLogger/success',0):.2f}"+
                           f" r={self._logs_batch.get('VecEnvLogger/ep_reward',float('nan')):08.8g}"+
                           f" min_r={self._logs_batch.get('VecEnvLogger/min.ep_reward',float('nan')):08.8g}"+
                           f" max_r={self._logs_batch.get('VecEnvLogger/max.ep_reward',float('nan')):08.8g}")
                self._logs_batch = {}
                self._logs_batch_size = 0

        return observation, reward, terminated, truncated, infos