from typing import Union, Optional, List
import gymnasium as gym
import numpy as np
import os
import torch as th
from lr_gym.utils.buffers import BasicStorage
from lr_gym.utils.utils import evaluatePolicy
from lr_gym.utils.tensor_trees import stack_tensor_tree, unstack_tensor_tree, map_tensor_tree
import lr_gym.utils.dbg.ggLog as ggLog

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
        verbose: int = 1
    ):
        if not isinstance(eval_env, gym.Env):
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

    def _init_callback(self) -> None:
        # Create folders if needed
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

                def predict(obs):
                    obs_batch = stack_tensor_tree(src_trees=[obs])
                    obs_batch = map_tensor_tree(src_tree=obs_batch, func = lambda t: t.expand(16,21).to(device=self._model.device))
                    return self._model.predict(obs_batch, deterministic = self.deterministic)[0].cpu().numpy(), None

                results = evaluatePolicy(self.eval_env,
                                            model = None,
                                            episodes=self.n_eval_episodes,
                                            predict_func=predict)
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
