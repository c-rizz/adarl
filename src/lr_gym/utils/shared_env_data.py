import torch as th
import torch.multiprocessing as mp
from lr_gym.utils.spaces import gym_spaces
from lr_gym.utils.utils import numpy_to_torch_dtype_dict
import ctypes

def create_tensor_tree(batch_size : int, space : gym_spaces.Space, share_mem : bool):
    if isinstance(space, gym_spaces.Dict):
        r = {}
        for k, s in space.spaces.items():
            r[k] = create_tensor_tree(batch_size, s, share_mem)
    elif isinstance(space, gym_spaces.Box):
        t = th.zeros(size = (batch_size,)+space.shape, dtype = numpy_to_torch_dtype_dict[space.dtype], )
        if share_mem:
            t.share_memory_()
        return t
    else:
        raise RuntimeError(f"Unsupported space {space}")

class SharedEnvData():
    def __init__(self, observation_space : gym_spaces.Space, action_space : gym_spaces.Space, info_space : gym_spaces.Space, n_envs : int, timeout_s : float):
        self._observation_space = observation_space
        self._action_space = action_space
        self._info_space = info_space
        self._n_envs = n_envs
        self._timeout_s = timeout_s

        self._shared_obss = create_tensor_tree(n_envs, self._observation_space, True)
        self._shared_acts = create_tensor_tree(n_envs, self._action_space, True)
        self._shared_infs = create_tensor_tree(n_envs, self._info_space, True)
        self._shared_dones = th.zeros(size=(self._n_envs,), dtype=th.bool).share_memory_()
        self._shared_truncs = th.zeros(size=(self._n_envs,), dtype=th.bool).share_memory_()
        self._shared_rews = th.zeros(size=(self._n_envs,), dtype=th.float32).share_memory_()

        self._envs_waiting_num = mp.Value(ctypes.c_int32, lock=False)
        self._np_envs_waiting = mp.Condition()

    def set_waiting(self):
        with self._np_envs_waiting:
            self._envs_waiting_num.value = self._n_envs

    def is_waiting(self):
        with self._np_envs_waiting:
            return self._envs_waiting_num.value>0

    def fill_data(self, env_num, observation, action, done, truncated, reward, info):
        self._shared_rews[env_num] = reward
        self._shared_dones[env_num] = done
        self._shared_truncs[env_num] = truncated
        pass
        with self._np_envs_waiting:
            self._envs_waiting_num.value -= 1
            self._np_envs_waiting.notify_all()

    def wait(self):
        with self._np_envs_waiting:
            didnt_timeout = self._np_envs_waiting.wait_for(lambda: self._envs_waiting_num.value>0, timeout=self._timeout_s)
        if not didnt_timeout:
            raise RuntimeError(f"SharedEnvData wait timed out")

def task(sh):
    pass

if __name__ == "__main__":
    sh = SharedEnvData(observation_space=gym_spaces.Dict({"obs",gym_spaces.Box(low = np.array([-1,-1]), high=np.array([1.0,1.0]))})
                       action_space=gym_spaces.Box(low = np.array([-1,-1]), high=np.array([1.0,1.0])),
                       info_space=gym_spaces.Dict({"info",gym_spaces.Box(low = np.array([-1,-1]), high=np.array([1.0,1.0]))}),
                    n_envs=1,timeout_s=60)

    worker = mp.Process(target=task, args=(sh,))