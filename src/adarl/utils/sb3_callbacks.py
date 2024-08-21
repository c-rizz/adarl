from __future__ import annotations
import os
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
import time
import adarl.utils.dbg.ggLog as ggLog
import numpy as np
import gymnasium as gym
from typing import Union, Optional, Dict, Any
from stable_baselines3.common.evaluation import evaluate_policy
import warnings
import adarl.utils.utils
import adarl.utils.session
import adarl.utils.sigint_handler
import torch as th

class CheckpointCallbackRB(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    .. warning::
      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``
    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0,
                       save_replay_buffer : bool = False,
                       save_freq_ep : int | None = None,
                       save_best = True):
        super(CheckpointCallbackRB, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_freq_ep = save_freq_ep
        self._last_saved_replay_buffer_path = None

        self._step_last_model_checkpoint = 0
        self._step_last_replay_buffer_checkpoint = 0
        self._episode_counter = 0
        self._save_best = save_best

        self._successes = [0]*50
        self._success_ratio = 0.0
        self._best_success_ratio = 0
        self._ep_last_model_checkpoint = 0


    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        # ggLog.info(f"AutoencodingSAC_VideoSaver: on_step, locals = {self.locals}")
        if self.locals["dones"][0]:
            self._episode_counter += 1
            info = self.locals["infos"][0]
            if "success" in info:
                ep_succeded = info["success"]
            else:
                ep_succeded = False
            self._successes[self._episode_counter%len(self._successes)] = int(ep_succeded)
            self._success_ratio = sum(self._successes)/len(self._successes)
        return True

    def _save_model(self, is_best, count_ep):
        self._best_success_ratio = max(self._best_success_ratio, self._success_ratio)
        if is_best:
            path = os.path.join(self.save_path, f"best_{self.name_prefix}_{self._episode_counter}_{self.model.num_timesteps}_{int(self._success_ratio*100)}_steps")
        else:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self._episode_counter}_{self.model.num_timesteps}_{int(self._success_ratio*100)}_steps")
        self.model.save(path)
        if not is_best:
            if count_ep:
                self._ep_last_model_checkpoint = self._episode_counter
            else:
                self._step_last_model_checkpoint = self.model.num_timesteps
        if self.verbose > 1:
            print(f"Saved model checkpoint to {path}")
        
        if self.save_replay_buffer:
            self._save_replay_buffer(is_best)

    def _save_replay_buffer(self, is_best):
        if is_best:
            path = os.path.join(self.save_path, f"best_{self.name_prefix}_replay_buffer_{self._episode_counter}_{self.model.num_timesteps}_steps")+".pkl"
        else:
            path = os.path.join(self.save_path, f"{self.name_prefix}_replay_buffer_{self._episode_counter}_{self.model.num_timesteps}_steps")+".pkl"
        t0 = time.monotonic()
        ggLog.info(f"Saving replay buffer with transitions {self.model.replay_buffer.size()}/{self.model.replay_buffer.buffer_size}...")
        self.model.save_replay_buffer(path)
        filesize_mb = os.path.getsize(path)/1024/1024
        if self._last_saved_replay_buffer_path is not None:
            os.remove(self._last_saved_replay_buffer_path) 
        t1 = time.monotonic()
        self._last_saved_replay_buffer_path = path
        self._step_last_replay_buffer_checkpoint = self.model.num_timesteps
        ggLog.debug(f"Saved replay buffer checkpoint to {path}, size = {filesize_mb}MB, transitions = {self.model.replay_buffer.size()}, took {t1-t0}s")

    def _on_rollout_end(self) -> bool:

        # done_array = np.array(self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones"))
        # if len(done_array)>1:
        #     raise RuntimeError("Only non-vectorized envs are supported.")
        # done = done_array.item()
        envsteps = self.model.num_timesteps
        if self._success_ratio > self._best_success_ratio and self._save_best:
            self._save_model(is_best=True, count_ep=False)
        if self.save_freq is not None and int(self.model.num_timesteps / self.save_freq) != int(self._step_last_model_checkpoint / self.save_freq):
            self._save_model(is_best=False, count_ep=False)
        if self.save_freq_ep is not None and int(self._episode_counter / self.save_freq_ep) != int(self._ep_last_model_checkpoint / self.save_freq_ep):
            self._save_model(is_best=False, count_ep=True)
        return True
    



class SigintHaltCallback(BaseCallback):

    def _on_step(self):
        return True
    
    def _on_rollout_end(self) -> bool:
        adarl.utils.sigint_handler.haltOnSigintReceived()
        return True
    


class PrintLrRunInfo(BaseCallback):

    def __init__(self, verbose: int = 0, print_freq_ep = 1):
        super().__init__(verbose=verbose)
        self._print_freq_ep = print_freq_ep
        self._episode_counter = 0
        self._step_counter = 0
        self._last_print_ep = -1

    
    def _on_step(self):
        self._step_counter += len(self.locals["dones"])
        dones_sum = sum(self.locals["dones"])
        if dones_sum:
            self._episode_counter += dones_sum
        adarl.utils.session.default_session.run_info["collected_episodes"].value = self._episode_counter
        adarl.utils.session.default_session.run_info["collected_steps"].value = self._episode_counter
        return True
    
    def _on_rollout_end(self) -> bool:
        if self._episode_counter - self._last_print_ep >=self._print_freq_ep:
            self._last_print_ep = self._episode_counter
            i = adarl.utils.session.default_session.run_info
            ggLog.info(f"{i['experiment_name']}:{i['run_id']} '{i['comment']}' eps={self._episode_counter} stps={self._step_counter}")
        return True

    

import adarl.utils.callbacks


class EvalCallback_ep(BaseCallback):
    """
    Callback for evaluating an agent.
    .. warning::
      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``
    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        eval_freq_ep: int = 10,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(verbose=verbose)
        self._sub_callback = adarl.utils.callbacks.EvalCallback(eval_env=eval_env,
                                                                 model=None,
                                                                 n_eval_episodes=n_eval_episodes,
                                                                 eval_freq_ep=eval_freq_ep,
                                                                 best_model_save_path=best_model_save_path,
                                                                 deterministic=deterministic,
                                                                 verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq_ep = eval_freq_ep
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        self.best_model_save_path = best_model_save_path
        self._episode_counter = 0
        self._step_counter = 0
        self._new_episode_counter = 0
        self._new_step_counter = 0


    def _init_callback(self) -> None:
        self._sub_callback.set_model(self.model)
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self._sub_callback.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self._sub_callback.eval_env}")


    def _on_step(self):
        # ggLog.info(f"AutoencodingSAC_VideoSaver: on_step, locals = {self.locals}")
        if self.locals["dones"][0]:
            self._episode_counter += 1
            self._new_episode_counter += 1
        n_envs = len(self.locals["dones"])
        self._step_counter += n_envs
        self._new_step_counter += n_envs
        return True

    def _on_rollout_end(self) -> bool:
        self._sub_callback.on_collection_end(collected_episodes=self._new_episode_counter,
                                              collected_steps=self._new_step_counter,
                                              collected_data=None)
        self._new_episode_counter = 0
        self._new_step_counter = 0
        return True
