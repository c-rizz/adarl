from typing import Union, Optional, List
import gymnasium as gym
import numpy as np
import os
import torch as th
from adarl.utils.buffers import BasicStorage
from adarl.utils.utils import evaluatePolicyVec
from adarl.utils.tensor_trees import stack_tensor_tree, unstack_tensor_tree, map_tensor_tree
import adarl.utils.dbg.ggLog as ggLog
import time

class TrainingCallback():

    def on_training_start(self):
        pass
    
    def on_collection_start(self):
        pass

    def on_collection_end(self,    collected_episodes : int,
                                    collected_steps : int,
                                    collected_data : Optional[BasicStorage] = None):
        pass

    def on_training_end(self):
        pass

class CallbackList(TrainingCallback):
    def __init__(self, callbacks : List[TrainingCallback]):
        self._callbacks = callbacks

    def on_training_start(self):
        for c in self._callbacks:
            c.on_training_start()
    
    def on_collection_start(self):
        for c in self._callbacks:
            c.on_collection_start()

    def on_collection_end(self,    collected_episodes : int,
                                    collected_steps : int,
                                    collected_data : Optional[BasicStorage] = None):
        for c in self._callbacks:
            c.on_collection_end(collected_episodes=collected_episodes,
                                collected_steps=collected_steps,
                                collected_data=collected_data)

    def on_training_end(self):
        
        for c in self._callbacks:
            c.on_training_end()

class EvalCallback(TrainingCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        model,
        n_eval_episodes: int = 10,
        eval_freq_ep: int = 10,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        verbose: int = 1,
        eval_name : str = "eval"
    ):
        if not isinstance(eval_env, gym.vector.VectorEnv):
            eval_env = gym.vector.SyncVectorEnv([lambda: eval_env], copy = False)
            raise NotImplementedError(f"eval_env can only be a gym.Env for now")
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq_ep = eval_freq_ep
        self.deterministic = deterministic
        self.verbose = verbose
        self._model = model
        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path

        self._episode_counter = 0
        self._last_evaluation_episode = float("-inf")
        self.best_mean_reward = float("-inf")
        self._episode_counter = 0
        self._step_counter = 0
        self.eval_name = eval_name

        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

    def set_model(self, model):
        self._model = model
    
    def on_collection_end(self,    collected_episodes : int,
                                    collected_steps : int,
                                    collected_data : Optional[BasicStorage] = None):
        self._episode_counter += collected_episodes
        self._step_counter += collected_steps
        # ggLog.info(f"on_collection_end: {self.eval_freq_ep} {self._episode_counter} {self._last_evaluation_episode}")
        if self.eval_freq_ep > 0 and self._episode_counter - self._last_evaluation_episode >= self.eval_freq_ep and self._last_evaluation_episode != self._episode_counter:
            # ggLog.info(f"Evaluating")

            cuda_sync_debug_state = th.cuda.get_sync_debug_mode()
            th.cuda.set_sync_debug_mode("default")
            try:
                self._last_evaluation_episode = self._episode_counter

                # def predict(obs):
                #     ggLog.info(f"Got obs of size {map_tensor_tree(obs, func = lambda t: t.size())}")
                #     obs_batch = stack_tensor_tree(src_trees=[obs])
                #     ggLog.info(f"Stacked obs to size {map_tensor_tree(obs, func = lambda t: t.size())}")
                #     # obs_batch = map_tensor_tree(src_tree=obs_batch, func = lambda t: t.expand(16,21).to(device=self._model.device))
                #     action, hidden_state = self._model.predict(obs_batch, deterministic = self.deterministic)
                #     ggLog.info(f"Returning action {action}, hidden state {hidden_state}")
                #     return action, hidden_state
                
                if self.verbose > 1:
                    ggLog.info(f"Evaluation '{self.eval_name}':")
                results = evaluatePolicyVec(self.eval_env,
                                            model = self._model,
                                            episodes=self.n_eval_episodes,
                                            deterministic=self.deterministic)
                mean_reward = results["reward_mean"]
                std_reward = results["reward_std"]
                mean_ep_length = results["steps_mean"]
                std_ep_length = results["steps_std"]
                self.last_mean_reward = mean_reward

                if self.verbose > 0:
                    print(f"Eval:" f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                    print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                
                if mean_reward > self.best_mean_reward:
                    if self.verbose > 0:
                        print("New best mean reward!")
                    if self.best_model_save_path is not None:
                        self._model.save(os.path.join(self.best_model_save_path, "best_model"))
                    self.best_mean_reward = mean_reward
            finally:
                th.cuda.set_sync_debug_mode(cuda_sync_debug_state)
        return True

class CheckpointCallbackRB(TrainingCallback):
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

    def __init__(self,  save_path: str, 
                        model,
                        buffer = None, 
                        name_prefix: str = "rl_model",
                        save_replay_buffer : bool = False,
                        save_freq: Optional[int] = None,
                        save_freq_ep : Optional[int] = None,
                        save_best = True):
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
        self._step_counter = 0

        self._successes = [0]*50
        self._success_ratio = 0.0
        self._best_success_ratio = 0
        self._ep_last_model_checkpoint = 0

        self._model = model
        self._buffer = buffer

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _save_model(self, is_best, count_ep):
        self._best_success_ratio = max(self._best_success_ratio, self._success_ratio)
        if is_best:
            path = os.path.join(self.save_path, f"best_{self.name_prefix}_{self._episode_counter}_{self._step_counter}_{int(self._success_ratio*100)}_steps")
        else:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self._episode_counter}_{self._step_counter}_{int(self._success_ratio*100)}_steps")
        self._model.save(path)
        if not is_best:
            if count_ep:
                self._ep_last_model_checkpoint = self._episode_counter
            else:
                self._step_last_model_checkpoint = self._step_counter
        
        if self.save_replay_buffer:
            self._save_replay_buffer(is_best)

    def _save_replay_buffer(self, is_best):
        if is_best:
            path = os.path.join(self.save_path, f"best_{self.name_prefix}_replay_buffer_{self._episode_counter}_{self._step_counter}_steps")+".pkl"
        else:
            path = os.path.join(self.save_path, f"{self.name_prefix}_replay_buffer_{self._episode_counter}_{self._step_counter}_steps")+".pkl"
        t0 = time.monotonic()
        if self._buffer is not None:
            ggLog.info(f"Saving replay buffer with transitions {self._buffer.replay_buffer.size()}/{self._buffer.replay_buffer.buffer_size}...")
            self._buffer.save(path)

        filesize_mb = os.path.getsize(path)/1024/1024
        if self._last_saved_replay_buffer_path is not None:
            os.remove(self._last_saved_replay_buffer_path) 
        t1 = time.monotonic()
        self._last_saved_replay_buffer_path = path
        self._step_last_replay_buffer_checkpoint = self._step_counter
        ggLog.debug(f"Saved replay buffer checkpoint to {path}, size = {filesize_mb}MB, transitions = {self._buffer.replay_buffer.size()}, took {t1-t0}s")

    def on_collection_end(self,    collected_episodes : int,
                                    collected_steps : int,
                                    collected_data : Optional[BasicStorage] = None):

        self._episode_counter += collected_episodes
        self._step_counter += collected_steps

        # TODO: update success_ratio using collected_data

        if self._success_ratio > self._best_success_ratio and self._save_best:
            self._save_model(is_best=True, count_ep=False)
        if self.save_freq is not None and self._step_counter - self._step_last_model_checkpoint >= self.save_freq:
            self._save_model(is_best=False, count_ep=False)
        if self.save_freq_ep is not None and self._episode_counter - self._ep_last_model_checkpoint >= self.save_freq_ep:
            self._save_model(is_best=False, count_ep=True)
        return True
    