from abc import ABC, abstractmethod
import adarl.utils.spaces as spaces
import numpy as np
import torch as th
from typing import final, TypeVar, Mapping, Generic
from gymnasium.vector.utils.spaces import batch_space
import adarl.utils.dbg.ggLog as ggLog
from adarl.utils.utils import masked_assign

State = Mapping[str | tuple[str,...], th.Tensor]
Observation = TypeVar("Observation", bound=Mapping[str | tuple[str,...], th.Tensor])
Action = th.Tensor

class BaseVecEnv(ABC, Generic[Observation]):
    def __init__(self,  num_envs : int,
                        single_action_space : spaces.gym_spaces.Space,
                        single_observation_space : spaces.gym_spaces.Space,
                        single_state_space : spaces.gym_spaces.Space,
                        info_space : spaces.gym_spaces.Space,
                        th_device : th.device,
                        single_reward_space = spaces.gym_spaces.Box(low=np.array([float("-inf")]), high=np.array([float("+inf")]), dtype=np.float32),
                        metadata = {},
                        max_episode_steps : int | th.Tensor = 1000,
                        seed : int = 0,
                        obs_dtype : th.dtype = th.float32,
                        build_and_initialize_ep : bool = False):
        self.num_envs = num_envs
        self._input_seed = seed
        self._rng = th.Generator(device=th_device)
        self._rng.manual_seed(seed)
        self._th_device = th_device
        self._obs_dtype = obs_dtype
        self.single_action_space = single_action_space
        self.single_observation_space = single_observation_space
        self.single_state_space = single_state_space
        self.single_reward_space = single_reward_space
        self.info_space = info_space
        self.vec_action_space = batch_space(single_action_space, n=num_envs)
        self.vec_observation_space = batch_space(single_observation_space, n=num_envs)
        self.vec_state_space = batch_space(single_state_space, n=num_envs)
        self.vec_reward_space = batch_space(single_reward_space, n=num_envs)

        self.metadata = metadata
        if isinstance(max_episode_steps, (int, float)):
            max_episode_steps = th.full(fill_value=max_episode_steps, size=(num_envs,),dtype=th.long, device=th_device)
        self._max_ep_steps = max_episode_steps
        self._tot_step_counter = 0
        self._ep_step_counter = th.zeros(size=(num_envs,), device=th_device, dtype=th.long)
        self._ep_counter = th.full(size=(num_envs,), fill_value=-1, device=th_device, dtype=th.long)
        self._tot_init_counter = 0
        self._init_counter_since_reset = 0

        if build_and_initialize_ep:
            self._build()
            self.initialize_episodes()

    def initialize_episodes(self, vec_mask : th.Tensor | None = None, options : dict = {}):
        """ Initialize the specified environments to an initial state. Internally it calls self._initialize_episodes().
            It also increments the proper counters (self._ep_counter).
            This method must not be overridden in the environment definition. Override _initialize_episodes() instead.

        Parameters
        ----------
        vec_mask : th.Tensor | None, optional
            Boolean tensor of shape (num_envs,) indicating which vectorized environments to initialize,
             by default None (which initializes all environments)
        options : dict
            Custom options for the initialization, content is the defined by each environment
        """
        if vec_mask is not None:
            # ggLog.info(f"self._ep_counter.device={self._ep_counter.device}, vec_mask.device={vec_mask.device}")
            masked_assign(self._ep_counter, vec_mask, self._ep_counter+1)
            masked_assign(self._ep_step_counter, vec_mask, 0)
            # th.where(vec_mask,self._ep_counter+1,self._ep_counter, out=self._ep_counter)
            # th.where(vec_mask,th.as_tensor(0, device=self._ep_step_counter.device),self._ep_step_counter, out=self._ep_step_counter)
            # self._ep_counter[vec_mask] += 1
            # self._ep_step_counter[vec_mask] = 0
        else:
            self._ep_counter += 1
            self._ep_step_counter.fill_(0)
        self._tot_init_counter += 1
        self._init_counter_since_reset += 1
        self._initialize_episodes(vec_mask, options)

    def get_ep_step_counter(self):
        return self._ep_step_counter

    @abstractmethod
    def _initialize_episodes(self, vec_mask : th.Tensor | None = None, options : dict = {}):
        """ Initialize the environments to an initial state. Called by initialize_episodes.
            Must be implemented in the environment definition.

        Parameters
        ----------
        vec_mask : th.Tensor | None, optional
            Boolean tensor of shape (num_envs,) indicating which vectorized environments to initialize,
             by default None (which initializes all environments)
        options : dict
            Custom options for the initialization, content is the defined by each environment
        """
        ...

    @abstractmethod
    def _build(self):
        """ Sets up the simulated/real scenario environment. Called only once at the construction of the environment.
        """
        ...

    
    def reset(self):
        """Re-initializes the scenario state to the initial state it had at build time and initializes. 
        """
        self._init_counter_since_reset = 0

    @abstractmethod
    def get_states(self) -> dict[str, th.Tensor]:
        ...

    @abstractmethod
    def get_observations(self, states : State) -> Observation:
        ...

    @abstractmethod
    def submit_actions(self, actions : th.Tensor):
        ...

    def on_step(self):
        """ Steps the environment forward of one step.
        """
        pass

    def step(self):
        """ Steps the environment forward of one step. Custom environments hould not override this method, 
            override on_step() instead if needed (most of the times you shouldn't need to).
        """
        th.add(self._ep_step_counter,1,out=self._ep_step_counter)
        self._tot_step_counter+=1
        self.on_step()

    @abstractmethod
    def get_times_since_build(self) -> th.Tensor:
        """ Environment time since episode start. Simulated time if it is a simulation.

        Returns
        -------
        th.Tensor
            Tensor of size (num_envs,) containing the time
        """
        ...

    @abstractmethod
    def get_times_since_ep_start(self) -> th.Tensor:
        """ Environment time since episode start. Simulated time if it is a simulation.

        Returns
        -------
        th.Tensor
            Tensor of size (num_envs,) containing the time in each simulation.
        """
        ...

    @abstractmethod
    def are_states_timedout(self, states : State) -> th.Tensor:
        """ Tells if the provided states represent environments that have timed out. 
            These are "truncated" episodes in OpenAI gym terminology

        Parameters
        ----------
        states : _type_
            Vectorized state in the format returned by get_states()

        Returns
        -------
        th.Tensor
            Boolean tensor of size (num_envs,)
        """
    

    @abstractmethod
    def are_states_terminal(self, states : State) -> th.Tensor:
        """ Tells if the provided states represent environments that have reached termination. 
            These are "terminated" episodes in OpenAI gym terminology

        Parameters
        ----------
        states : _type_
            Vectorized state in the format returned by get_states()

        Returns
        -------
        th.Tensor
            Boolean tensor of size (num_envs,)
        """
    
    @abstractmethod
    def compute_rewards(self, states : State, sub_rewards_return : dict | None = None) -> th.Tensor:
        """ Computes the rewards for the provided state.
            Can also return some extra information in extra_info_return

        Parameters
        ----------
        state : _type_
            _description_
        sub_rewards_return : dict | None, optional
            _description_, by default None

        Returns
        -------
        th.Tensor
            Tensor of shape (num_envs,) containing the reward for each state
        """
        ...

    @abstractmethod
    def get_ui_renderings(self, vec_mask : th.Tensor) -> tuple[list[th.Tensor], th.Tensor]:
        """Get environment renderings for user visualization.

        Parameters
        ----------
        vec_mask : th.Tensor
            Boolean tensor of shape (num_envs,) indicating which vectorized environments to render.

        Returns
        -------
        tuple[list[th.Tensor], th.Tensor]
            A tuple with a batch of batches of images in the first element and the sim time of each image
            in the second.
            The first element contains a list of length len(ui_cameras) containing tensors of shape
            (th.count_nonzero(vec_size), <image_shape>) and the second has shape(th.count_nonzero(vec_size), len(ui_cameras))
        """
        ...

    @abstractmethod
    def get_infos(self,state, labels : dict[str, th.Tensor] | None = None) -> dict[str, th.Tensor]:
        """ Gets environment specific extra information. The content is environment-defined, should not contain
            information actually used for the functioning of the environment, but just metrics or debug infos.

        Parameters
        ----------
        state : State
            States in the format used by get_states

        Returns
        -------
        dict[str, th.Tensor]
            _description_
        """
        ...

    def get_max_episode_steps(self) -> th.Tensor:
        """Gets the maximum episode duration for each single environment.

        Returns
        -------
        th.Tensor
            Tensor of size (num_envs,) and type th.long.
        """
        return self._max_ep_steps
    

    def set_max_episode_steps(self, max_episode_steps : th.Tensor):
        """Gets the maximum episode duration for each single environment.

        Returns
        -------
        th.Tensor
            Tensor of size (num_envs,) and type th.long.
        """
        self._max_ep_steps = max_episode_steps

    @abstractmethod
    def close(self):
        """ Close the environment, releasing any resource that was held by it.
        """
        ...

    def set_seeds(self, seeds : th.Tensor):
        self._rng_seeds = seeds.expand((self.num_envs,))
        self._rng.manual_seed(th.sum(self._rng_seeds).item())


    def get_seeds(self):
        return self._rng_seeds

    
    def _thtens(self, tensor):
        return th.as_tensor(tensor, dtype=self._obs_dtype).to(device=self._th_device, non_blocking=True)

    def _thzeros(self, size : tuple[int,...]):
        return th.zeros(size, dtype=self._obs_dtype).to(device=self._th_device, non_blocking=True)

    def _thrand(self, size : tuple[int,...]):
        return th.rand(size=size, dtype=self._obs_dtype, device=self._th_device, generator=self._rng)
    
    def _thrandn(self, size : tuple[int,...]):
        return th.randn(size=size, dtype=self._obs_dtype, device=self._th_device, generator=self._rng)