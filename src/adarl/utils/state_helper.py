from __future__ import annotations
from abc import ABC, abstractmethod
from adarl.utils.tensor_trees import TensorMapping
from enum import IntEnum, Enum
from typing import TypeVar, Sequence, Mapping, Any, SupportsFloat, Union, Tuple
from typing_extensions import override
import adarl.utils.dbg.ggLog as ggLog
import adarl.utils.spaces as spaces
import adarl.utils.tensor_trees
import adarl.utils.utils
import dataclasses
import numpy as np
import torch as th
import typing

_T = TypeVar('_T', float, th.Tensor)
def unnormalize(v : _T, min : _T, max : _T) -> _T:
    return min+(v+1)/2*(max-min)

def normalize(value : _T, min : _T, max : _T):
    return (value + (-min))/(max-min)*2-1


FieldName = Union[str, int, Tuple[str,str]]

class StateHelper(ABC):
    @abstractmethod
    def reset_state(self, initial_values = None):
        ...
    @abstractmethod
    def update(self, instantaneous_state, state):
        ...
    @abstractmethod
    def flatten(self, state) -> th.Tensor:
        ...
    @abstractmethod
    def flat_state_names(self) -> list[str]:
        ...

    @abstractmethod
    def normalize(self, state : _T) -> _T:
        ...
    
    @abstractmethod
    def unnormalize(self, state : _T) -> _T:
        ...
    
    @abstractmethod
    def observe(self, state):
        ...

    @abstractmethod
    def observation_names(self):
        ...
    
    @abstractmethod
    def get_space(self) -> spaces.gym_spaces.Space:
        ...
    
    @abstractmethod
    def get_obs_space(self) -> spaces.gym_spaces.Space:
        ...
    
    @abstractmethod
    def get(self, state, field_names : Sequence[FieldName]) -> th.Tensor:
        ...

    @abstractmethod
    def field_idx(self, field_names : Sequence[FieldName], device : th.device) -> th.Tensor:
        ...




class ThBoxStateHelper(StateHelper):
    def __init__(self,  field_names : Sequence[FieldName], obs_dtype : th.dtype,
                        th_device : th.device, fields_minmax : Mapping[FieldName,th.Tensor|Sequence[float]|Sequence[th.Tensor]],
                        field_size : list[int] | tuple[int,...], 
                        history_length : int = 1,
                        observable_fields : list[str|int] | None = None,
                        subfield_names : list[str] | np.ndarray | None = None,
                        flatten_observation = False,
                        obs_history_length : int = 1):
        self.field_names = field_names
        self.field_size = tuple(field_size)
        if subfield_names is not None:
            if isinstance(subfield_names,(list,tuple)):
                if len(self.field_size)!=1:
                    raise RuntimeError(f"subfield_names can be a list only if fields are 1-dimensional")
                if len(subfield_names)!=self.field_size[0]:
                    raise RuntimeError(f"subfield_names is not of size field_size[0], len(subfield_names)={len(subfield_names)} and field_size[0] is {self.field_size[0]}")
            elif isinstance(subfield_names, np.ndarray):
                if subfield_names.shape != self.field_size:
                    raise RuntimeError(f"subfield_names is not of size field_size, subfield_names.shape={subfield_names.shape} and field_size is {self.field_size}")
        if isinstance(subfield_names,(list,tuple)):
            subfield_names = np.array(subfield_names, dtype=object)
        if obs_history_length>history_length:
            raise RuntimeError(f"obs_history_length ({obs_history_length}) must be less than history_length ({history_length})")
        self.subfield_names = subfield_names
        self._obs_dtype = obs_dtype
        self._th_device = th_device
        self._history_length = history_length
        self._obs_history_length = obs_history_length
        self._observable_fields = field_names if observable_fields is None else observable_fields
        self._flatten_observation = flatten_observation
        self._obs_names = None
        self._state_names = None

        self._fully_observable = observable_fields is None and self._obs_history_length==self._history_length
        self._field_idxs = {n:field_names.index(n) for n in field_names}
        if self.subfield_names is not None:
            self._subfield_idxs = {self.subfield_names[idx]:idx for idx in np.ndindex(self.subfield_names.shape)}
        else:
            self._subfield_idxs = None
        self._fields_num = len(field_names)
        self._state_size = (self._history_length, self._fields_num) + self.field_size
        self._obs_size = (self._obs_history_length, len(self._observable_fields)) + self.field_size
        self._limits_minmax = self.build_limits(fields_minmax)
        assert self._limits_minmax.size() == (2, self._fields_num,)+self.field_size, f"failed {self._limits_minmax.size()} == {(2, self._fields_num,)+self.field_size}"

        self._observable_fields = [f for f in self.field_names if f in self._observable_fields] # to ensure they are ordered
        self._observable_indexes = th.as_tensor([self.field_names.index(n) for n in self._observable_fields], dtype=th.int32)
        # print(f"observable_indexes = {self._observable_indexes}")
        lmin = self._limits_minmax[0]
        hlmin = lmin.expand(self._history_length, *lmin.size())
        lmax = self._limits_minmax[1]
        hlmax = lmax.expand(self._history_length, *lmax.size())
        self._state_space = spaces.ThBox(low=hlmin, high=hlmax, shape=self._state_size)
        self._obs_space = spaces.ThBox(low=self.observe(hlmin), high=self.observe(hlmax), shape=self._obs_size, dtype=self._obs_dtype,
                                       labels=self.observation_names())

    def build_limits(self, fields_minmax : Mapping[FieldName,th.Tensor|Sequence[float]|Sequence[th.Tensor]]):
        new_minmax = {}
        for n, minmax in fields_minmax.items():
            if not isinstance(minmax, th.Tensor):
                minmax = th.as_tensor(minmax)
            minmax = minmax.squeeze()
            if minmax.size() == (2,):
                minmax = minmax.expand(self.field_size+(2,)).permute(-1,*range(len(self.field_size)))
            if minmax.size()!=(2,)+self.field_size:
                raise RuntimeError(f"Field {n} has size {minmax.size()}, should be {(2,)+self.field_size}")
            new_minmax[n]=minmax
        fields_minmax = new_minmax
        return th.stack([th.as_tensor(fields_minmax[fn], dtype=self._obs_dtype, device=self._th_device) for fn in self.field_names]).transpose(0,1)

    def _mapping_to_tensor(self, instantaneous_state : Mapping[FieldName,th.Tensor | float | Sequence[float]]) -> th.Tensor:
        instantaneous_state = {k:th.as_tensor(v) for k,v in instantaneous_state.items()}
        instantaneous_state = {k:v.unsqueeze(0) if v.dim()==0 else v for k,v in instantaneous_state.items()}
        return th.stack([instantaneous_state[k] for k in self.field_names])

    @override
    def reset_state(self, initial_values : th.Tensor | SupportsFloat | Mapping[FieldName,th.Tensor | float | Sequence[float]] | None= None):
        if initial_values is None:
            initial_values = th.tensor(0.0)
        if isinstance(initial_values,Mapping):
            initial_values = self._mapping_to_tensor(initial_values)
        elif isinstance(initial_values,(SupportsFloat, Sequence)):
            initial_values = th.as_tensor(initial_values)
        if initial_values.numel() == 1:
            initial_values = th.full(   fill_value=initial_values.item(),
                                        size = self._state_size[1:],
                                        device=self._th_device,
                                        dtype=self._obs_dtype)
        if self._fields_num == 1 and initial_values.size() == self._state_size[2:]:
            initial_values = initial_values.unsqueeze(0)
        if initial_values.size()!=self._state_size[1:]:
            raise RuntimeError( f"Unexpected intial value shape {initial_values.size()}, should be {self._state_size[1:]}"
                                f" Fields are {self.field_names}, subfields are {self.subfield_names}")
        initial_values = initial_values.to(self._th_device, self._obs_dtype)
        state = initial_values.repeat(self._history_length, *((1,)*len(initial_values.size())))
        assert state.size() == self._state_size,    f"Unexpected resulting state size {state.size()}, should be {self._state_size}."\
                                                    f" Fields are {self.field_names}, subfields are {self.subfield_names}"
        return state
    
    @override
    def update(self, instantaneous_state : th.Tensor | Mapping[FieldName,th.Tensor | float | Sequence[float]], state : th.Tensor):
        if isinstance(instantaneous_state,Mapping):
            instantaneous_state = self._mapping_to_tensor(instantaneous_state)
        for i in range(1,state.size()[0]):
            state[i] = state[i-1]
        if self.field_size == (1,) and instantaneous_state.size() == (self._fields_num,):
            instantaneous_state = instantaneous_state.unsqueeze(1)
        state[0] = instantaneous_state
        return state
    
    @override
    def flatten(self, state : th.Tensor):
        return state.flatten()
    
    def flat_obs_names(self):
        ret = self.observation_names().flatten()
        # ggLog.info(f"flat_obs_names = {ret}")
        return ret
    
    @override
    def flat_state_names(self):
        return self.state_names().flatten()

    @override
    def normalize(self, state : th.Tensor, alternative_limits : th.Tensor | None = None, warn_limits_violation = False):
        limits = self._limits_minmax if alternative_limits is None else alternative_limits
        ret = normalize(state, limits[0], limits[1])
        if warn_limits_violation and th.any(th.abs(ret) > 1.1):
            ggLog.warn(f"Normalization exceeded [-1.1,1.1] range: {state} with {limits[0]} & {limits[1]} = {ret}")
        return ret
    
    @override
    def unnormalize(self, state : th.Tensor):
        return unnormalize(state, self._limits_minmax[0], self._limits_minmax[1])
    
    @override
    def observe(self, state : th.Tensor):
        if self._fully_observable:
            obs = state
        else:
            obs = state[:self._obs_history_length,self._observable_indexes]
        if self._flatten_observation:
            obs = th.flatten(obs)
        return obs
    
    @override
    def observation_names(self):
        if self._obs_names is None:
            self._obs_names = np.empty(shape=(self._obs_history_length,len(self._observable_fields))+self.field_size, dtype=object)
            for h in range(self._obs_history_length):
                for fn in range(len(self._observable_fields)):
                    for s in np.ndindex(self.field_size):
                        f = self._observable_fields[fn]
                        if isinstance(f, Enum):
                            f = f.name
                        if self.subfield_names is not None:
                            self._obs_names[(h,fn)+s] = f"[{h},{f},{self.subfield_names[s]}]"
                        else:
                            self._obs_names[(h,fn)+s] = f"[{h},{f},{','.join([str(i) for i in s])}]"
            if self._flatten_observation:
                self._obs_names =  self._obs_names.flatten()
        return self._obs_names
    
    def state_names(self):
        if self._state_names is None:
            self._state_names = np.empty(shape=(self._history_length,self._fields_num)+self.field_size, dtype=object)
            for h in range(self._history_length):
                for fn in range(self._fields_num):
                    for s in np.ndindex(self.field_size):
                        f = self.field_names[fn]
                        if isinstance(f, Enum):
                            f = f.name
                        if self.subfield_names is not None:
                            self._state_names[(h,fn)+tuple(s)] = f"[{h},{f},{self.subfield_names[s]}]"
                        else:
                            self._state_names[(h,fn)+tuple(s)] = f"[{h},{f},{','.join([str(i) for i in s])}]"
        return self._state_names
    

    @override
    def get_space(self):
        return self._state_space
    
    @override
    def get_obs_space(self):
        return self._obs_space
    
    @override
    def get(self, state : th.Tensor, field_names : Sequence[FieldName] | tuple[Sequence[FieldName],Sequence[FieldName]]):
        if len(field_names)>0 and isinstance(field_names[0],Sequence) and isinstance(field_names[1],Sequence):
            return state[:self.field_idx(field_names[0], device=state.device), self.subfield_idx(field_names[1], device=state.device)]
        else:
            return state[:,self.field_idx(field_names, device=state.device)]

    @override
    def field_idx(self, field_names : Sequence[FieldName], device : th.device):
        return th.as_tensor([self._field_idxs[n] for n in field_names], device=device)
    
    def subfield_idx(self, subfield_names : Sequence[FieldName], device : th.device):
        if self.subfield_names is not None:
            return th.as_tensor([self._subfield_idxs[n] for n in subfield_names], device=device)
        else:
            # Then subfield names are just the indexes
            return th.as_tensor([int(typing.cast(int, n)) for n in subfield_names], device=device)

    def get_limits(self):
        return self._limits_minmax

class StateNoiseGenerator:
    def __init__(self, state_helper : ThBoxStateHelper, generator : th.Generator, 
                        episode_mu_std : Mapping[FieldName,th.Tensor] | th.Tensor | list[float] | tuple[float],
                        step_std : Mapping[FieldName,th.Tensor] | th.Tensor | list[float] | tuple[float] | float,
                        dtype : th.dtype, device : th.device,
                        squash_sigma = 3):
        self._state_helper = state_helper
        self._field_names = state_helper.field_names
        self._fields_num = len(self._field_names)
        self._field_size = state_helper.field_size
        self._rng = generator
        self._device = device
        self._dtype = dtype
        self._history_length = state_helper._history_length
        self._noise_shape = state_helper.get_space().shape[1:]
        self._squash_sigma = squash_sigma
        if isinstance(episode_mu_std,(list,tuple,float)):
            episode_mu_std = th.as_tensor(episode_mu_std)
        if isinstance(step_std,(list,tuple,float)):
            step_std = th.as_tensor(step_std)
        if isinstance(episode_mu_std,th.Tensor):
            required_size = (2,self._fields_num)+self._field_size
            if episode_mu_std.size() == (2,):
                self._episode_mu_std = episode_mu_std.expand(*((self._fields_num,)+self._field_size+(2,))).permute(2,0,1)
            elif episode_mu_std.size() == ((2,)+self._field_size):
                self._episode_mu_std = episode_mu_std.unsqueeze(1).expand(*required_size)
            elif episode_mu_std.size() == required_size:
                self._episode_mu_std = episode_mu_std
            else:
                raise RuntimeError(f"Unexpected episode_mu_std size {episode_mu_std.size()}, should either be (2,) or {required_size}")
            self._episode_mu_std = self._episode_mu_std.to(dtype=self._dtype,device=self._device)
        elif isinstance(episode_mu_std,Mapping):
            self._episode_mu_std = th.as_tensor([episode_mu_std[k] for k in self._field_names], dtype=self._dtype, device=self._device).permute(1,0)

        if isinstance(step_std,th.Tensor):
            required_size = (self._fields_num,)+self._field_size
            if step_std.numel() == 1:
                self._step_std = step_std.expand(*((self._fields_num,)+self._field_size))
            elif step_std.size() == self._field_size:
                self._step_std = step_std.unsqueeze(0).expand(*required_size)
            elif step_std.size() == required_size:
                self._step_std = step_std
            else:
                raise RuntimeError(f"Unexpected step_std size {step_std.size()}, should either be (1,),(,) or {required_size}")
            self._step_std = self._step_std.to(dtype=self._dtype,device=self._device)
        elif isinstance(step_std,Mapping):
            self._step_std = th.as_tensor([step_std[k] for k in self._field_names], dtype=self._dtype, device=self._device).reshape(self._noise_shape)

        assert self._episode_mu_std.size() == (2,)+self._noise_shape
        assert self._step_std.size() == self._noise_shape

        # # ggLog.info(f"Noise generator got [{self._episode_mu_std},{self._step_std}]")
        state_limits = state_helper.get_limits()
        self._fields_scale = state_limits[1]-state_limits[0]
        # self._episode_mu_std = self._episode_mu_std*self._fields_scale.expand(2, *self._fields_scale.size())
        # self._step_std = self._step_std*self._fields_scale
        # # ggLog.info(f"Noise generator unnormalized to [{self._episode_mu_std},{self._step_std}]")


        # At the beginning of each episode a mu is sampled
        # then at each step the noise uses this mu and the step_std
        self._resample_mu()
        # At each step the noise state contains the current sampled noise
        self._state_space = spaces.ThBox(low=float("-inf"), high=float("+inf"), shape=(self._history_length,)+self._noise_shape)

    def get_space(self):
        return self._state_space

    def _resample_mu(self):
        self._current_ep_mustd = th.stack([adarl.utils.utils.randn_from_mustd(self._episode_mu_std,
                                                                              size = self._noise_shape,
                                                                              generator=self._rng,
                                                                              squash_sigma=self._squash_sigma),
                                           self._step_std.expand(self._noise_shape)])
        

    def _generate_noise(self):
        return adarl.utils.utils.randn_from_mustd(self._current_ep_mustd, generator=self._rng,
                                                  squash_sigma = self._squash_sigma)

    def reset_state(self):
        self._resample_mu()
        return th.stack([self._generate_noise() for _ in range(self._history_length)])
    
    def update(self, state):
        for i in range(1,state.size()[0]):
            state[i] = state[i-1]
        state[0] = self._generate_noise()
        return state

    def normalize(self, noise):
        return noise / self._fields_scale
    
    def unnormalize(self, noise):
        return noise*self._fields_scale


State = TypeVar("State", bound=Mapping)
class DictStateHelper(StateHelper):
    def __init__(self, state_helpers : dict[str, ThBoxStateHelper],
                 observable_fields : list[str] | None = None,
                 noise : Mapping[str,StateNoiseGenerator] = {},
                 flatten_in_obs : list[str] = [],
                 flattened_part_name : str = "vec"):
        # Would be nice to make this recursive (i.e. make this contain also DictStateHelpers), 
        # but it becomes a bit of a mess from the typing point of view
        self.sub_helpers = state_helpers
        self._flatten_in_obs = flatten_in_obs
        self._flatten_part_name = flattened_part_name
        for k in state_helpers:
            if k.endswith("_n"):
                raise RuntimeError(f"keys ending in '_n' are not allowed")
        self.noise_generators = noise
        for k in self.noise_generators:
            if k not in self.sub_helpers:
                raise RuntimeError(f"Received noise for state '{k}', but not state '{k}' was specified.")
        if observable_fields is None:
            observable_fields = list(state_helpers.keys())
        self._observable_fields = observable_fields
        subspaces = {k:s.get_space() for k,s in self.sub_helpers.items()}
        subspaces.update({k+"_n":s.get_space() for k,s in self.noise_generators.items()})
        self._state_space = spaces.gym_spaces.Dict(subspaces)
        obs_subspaces = {k:self.sub_helpers[k].get_obs_space() for k in self._observable_fields if k not in self._flatten_in_obs}
        if len(self._flatten_in_obs)>0:
            self._flattened_part_size = int(sum([np.prod(self.sub_helpers[k].get_obs_space().shape) for k in self._flatten_in_obs ]))
            flattened_dtype = self.sub_helpers[self._flatten_in_obs[0]].get_obs_space().dtype
            for k in self._flatten_in_obs:
                d = self.sub_helpers[k].get_obs_space().dtype
                if d != flattened_dtype:
                    raise RuntimeError(f"All sub observations that are flattened should have the same dtype, "
                                       f"but {self._flatten_in_obs[0]} has {flattened_dtype} and {k} has {d}")
            obs_subspaces[self._flatten_part_name] = spaces.ThBox(low = -1, high = 1, shape=(self._flattened_part_size,), dtype=flattened_dtype)
        self._obs_space = spaces.gym_spaces.Dict(obs_subspaces)
        
    def add_substate(self,  state_name : str,
                            state_helper : ThBoxStateHelper,
                            observable : bool,
                            flatten : bool,
                            noise : StateNoiseGenerator | None= None) -> DictStateHelper:
        state_helpers = {state_name:state_helper}
        state_helpers.update(self.sub_helpers)
        if observable:
            observable_fields = [state_name]
        else:
            observable_fields = []
        if flatten:
            flatten_in_obs = [state_name]
        else:
            flatten_in_obs = []
        flatten_in_obs.extend(self._flatten_in_obs)
        observable_fields.extend(self._observable_fields)
        if noise is not None:
            noises = {state_name:noise}
        else:
            noises = {}
        noises.update(self.noise_generators)        
        return DictStateHelper( state_helpers=state_helpers,
                                observable_fields=observable_fields,
                                flatten_in_obs=flatten_in_obs,
                                flattened_part_name=self._flatten_part_name,
                                noise=noises)
        
    
    @override
    def reset_state(self, initial_values: State | None = None) -> State:
        if initial_values is None:
            initial_values = {k:th.tensor(0.0) for k in self.sub_helpers.keys()}
        state = {k:self.sub_helpers[k].reset_state(v) for k,v in initial_values.items()}
        noise_state = {k+"_n":ng.reset_state() for k,ng in self.noise_generators.items()}
        state.update(noise_state)
        return state        
    
    @override
    def update(self, instantaneous_state : Mapping[str,th.Tensor | Mapping[FieldName, th.Tensor]], state : Mapping[str,th.Tensor]):
        for k,sh in self.sub_helpers.items():
            sh.update(instantaneous_state[k], state[k])
        for k,ng in self.noise_generators.items():
            ng.update(state[k+"_n"])        

    @override
    def normalize(self, state : dict[str,th.Tensor]):
        ret = {k:sh.normalize(state[k]) for k,sh in self.sub_helpers.items()}
        ret.update({k+"_n":ng.normalize(state[k+"_n"]) for k,ng in self.noise_generators.items()})
        return ret
    
    @override
    def unnormalize(self, state : dict[str,th.Tensor]):
        ret = {k:sh.unnormalize(state[k]) for k,sh in self.sub_helpers.items()}
        ret.update({k+"_n":ng.unnormalize(state[k+"_n"]) for k,ng in self.noise_generators.items()})
        return ret

    @override
    def observe(self, state:  dict[str,th.Tensor]):
        # ggLog.info(f"observing state {state}")
        state = self.normalize(state)
        # ggLog.info(f"normalized state = {state}")
        noisy_state = {k:ss+state[k+"_n"] if k in self.noise_generators else ss for k,ss in state.items()}
        nonflat_obs = {k:self.sub_helpers[k].observe(noisy_state[k]) for k in  self._observable_fields}
        # ggLog.info(f"non_flat_obs = {nonflat_obs}")
        flattened_parts = []
        obs = {}
        for k,subobs in nonflat_obs.items():
            if k in self._flatten_in_obs:
                flattened_parts.append(self.sub_helpers[k].flatten(subobs))
            else:
                obs[k] = subobs
        if len(flattened_parts) > 0:
            obs[self._flatten_part_name] = th.concat(flattened_parts)
            # if th.any(th.abs(obs[self._flatten_part_name]) > 1.0):
            #     ggLog.warn(f"observation values exceed -1,1 normalization: nonflat_obs = {nonflat_obs},\nstate = {state}")
        return obs

    @override    
    def observation_names(self) -> dict[str,np.ndarray]:
        flattened_parts_names = []
        obs_names = {}
        for k in self._observable_fields:
            if k in self._flatten_in_obs:
                flattened_parts_names.extend([k+"_"+str(n) for n in self.sub_helpers[k].flat_obs_names()])
            else:
                obs_names[k] = self.sub_helpers[k].observation_names()
        if len(flattened_parts_names) > 0:
            # ggLog.info(f"flattened_parts_names = {flattened_parts_names}")
            obs_names[self._flatten_part_name] = np.array(flattened_parts_names)
        return obs_names

    
    @override
    def get_space(self):
        return self._state_space
    
    @override
    def get_obs_space(self):
        return self._obs_space
    
    @override
    def get(self, state : dict[str,th.Tensor],
            field_names : Sequence[tuple[str,Sequence[FieldName] | tuple[Sequence[FieldName],Sequence[FieldName]]]]):
        return [self.sub_helpers[k].get(state[k], idxs) for k,idxs in field_names]
    
    def get_t(self, state : dict[str,th.Tensor], field_names : tuple[str,Sequence[str|int]]):
        ss = field_names[0]
        return self.sub_helpers[ss].get(state[ss], field_names=field_names[1])
    
    @override
    def field_idx(self, field_names: dict[str,list[str | int]], device: th.device):
        return {k:self.sub_helpers[k].field_idx(idxs, device=device) for k,idxs in field_names.items()}
    
    @override
    def flatten(self, state : dict[str,th.Tensor], include_only : list[str] | None = None):
        rets = []
        for k,sh in self.sub_helpers.items():
            if include_only is None or k in include_only:
                rets.append(sh.flatten(state[k]))
        return th.concat(rets)
    
    @override
    def flat_state_names(self, include_only : list[str] | None = None):
        rets = []
        for k,sh in self.sub_helpers.items():
            if include_only is None or k in include_only:
                rets.extend([f"{k}.{sn}" for sn in sh.flat_state_names()])
        return rets



class RobotStateHelper(ThBoxStateHelper):
    def __init__(self,  joint_limit_minmax_pve : Mapping[tuple[str,str],np.ndarray | th.Tensor],
                        stiffness_minmax : tuple[float,float] | Mapping[tuple[str,str],np.ndarray | th.Tensor],
                        damping_minmax : tuple[float,float] | Mapping[tuple[str,str],np.ndarray | th.Tensor],
                        obs_dtype : th.dtype,
                        th_device : th.device,
                        history_length : int = 1,
                        obs_history_length : int = 1):
        subfield_names = ["pos","vel","eff","refpos","refvel","refeff","stiff","damp"]
        self._th_device = th_device
        super().__init__(   field_names=list(joint_limit_minmax_pve.keys()),
                            obs_dtype=obs_dtype,
                            th_device=th_device,
                            field_size=(len(subfield_names),),
                            fields_minmax= self._build_fields_minmax(joint_limit_minmax_pve, stiffness_minmax, damping_minmax),
                            history_length=history_length,
                            subfield_names = subfield_names,
                            obs_history_length=obs_history_length)

    def _build_fields_minmax(self,  joint_limit_minmax_pve : Mapping[tuple[str,str],np.ndarray | th.Tensor],
                                    stiffness_minmax : tuple[float,float] | Mapping[tuple[str,str],np.ndarray | th.Tensor],
                                    damping_minmax : tuple[float,float] | Mapping[tuple[str,str],np.ndarray | th.Tensor]) -> Mapping[FieldName,th.Tensor|Sequence[float]|Sequence[th.Tensor]]:
        if isinstance(stiffness_minmax,tuple):
            stiffness_minmax = {j:th.as_tensor(stiffness_minmax, device=self._th_device) for j in joint_limit_minmax_pve}
        if isinstance(damping_minmax,tuple):
            damping_minmax = {j:th.as_tensor(damping_minmax, device=self._th_device) for j in joint_limit_minmax_pve}
        t_joint_limit_minmax_pve : Mapping[Any, th.Tensor] = adarl.utils.tensor_trees.map_tensor_tree(joint_limit_minmax_pve, lambda v: th.as_tensor(v,device = self._th_device)) #type: ignore
        t_stiffness_minmax : Mapping[Any, th.Tensor] = adarl.utils.tensor_trees.map_tensor_tree(stiffness_minmax, lambda v: th.as_tensor(v,device = self._th_device)) #type: ignore
        t_damping_minmax : Mapping[Any, th.Tensor] = adarl.utils.tensor_trees.map_tensor_tree(damping_minmax, lambda v: th.as_tensor(v,device = self._th_device)) #type: ignore
        return {joint : th.stack([  limits_minmax_pve[:,0],
                                    limits_minmax_pve[:,1],
                                    limits_minmax_pve[:,2],
                                    limits_minmax_pve[:,0],
                                    limits_minmax_pve[:,1],
                                    limits_minmax_pve[:,2],
                                    t_stiffness_minmax[joint],
                                    t_damping_minmax[joint]]).permute(1,0)
                for joint,limits_minmax_pve in t_joint_limit_minmax_pve.items()}
        
    def build_robot_limits(self, joint_limit_minmax_pve : Mapping[tuple[str,str],np.ndarray | th.Tensor],
                            stiffness_minmax : tuple[float,float] | Mapping[tuple[str,str],np.ndarray | th.Tensor],
                            damping_minmax : tuple[float,float] | Mapping[tuple[str,str],np.ndarray | th.Tensor]):
        return super().build_limits(fields_minmax=self._build_fields_minmax(joint_limit_minmax_pve, stiffness_minmax, damping_minmax))


    def state_names(self):
        if self._state_names is None:
            self._state_names = np.empty(shape=(self._history_length,self._fields_num)+self.field_size, dtype=object)
            for h in range(self._history_length):
                for fn in range(self._fields_num):
                    for s in np.ndindex(self.field_size):
                        jname = self.field_names[fn][1]
                        self._state_names[(h,fn)+tuple(s)] = f"[{h},{jname},{self.subfield_names[s]}]"
        return self._state_names

class RobotStatsStateHelper(ThBoxStateHelper):
    def __init__(self,  joint_limit_minmax_pve : Mapping[tuple[str,str],np.ndarray | th.Tensor],
                        obs_dtype : th.dtype,
                        th_device : th.device,
                        history_length : int = 1):
        subfield_names = [  "minpos","minvel","minacc","mineff",
                            "maxpos","maxvel","maxacc","maxeff",
                            "avgpos","avgvel","avgacc","avgeff",
                            "stdpos","stdvel","stdacc","stdeff"]
        joint_limit_minmax_pve = {k:th.as_tensor(v) for k,v in joint_limit_minmax_pve.items()}
        jlims_minmax_pvae = {jn:th.cat([minmax_pve[:,:2],
                                        th.stack([minmax_pve[0,1]-minmax_pve[1,1], minmax_pve[1,1]-minmax_pve[0,1]]).unsqueeze(1),
                                        minmax_pve[:,[2]]], dim=1) for jn,minmax_pve in joint_limit_minmax_pve.items()}
        super().__init__(   field_names = list(jlims_minmax_pvae.keys()),
                            obs_dtype = obs_dtype,
                            th_device = th_device,
                            field_size = (len(subfield_names),),
                            fields_minmax= self._build_fields_minmax(jlims_minmax_pvae),
                            history_length = history_length,
                            subfield_names = subfield_names)

    def _build_fields_minmax(self,  joint_limit_minmax_pve : Mapping[tuple[str,str],np.ndarray | th.Tensor]
                             ) -> Mapping[FieldName,th.Tensor|Sequence[float]|Sequence[th.Tensor]]:
        ret = {}
        for joint,limits_minmax_pve in joint_limit_minmax_pve.items():
            limits_minmax_pve = th.as_tensor(limits_minmax_pve)
            if limits_minmax_pve.size() != (2,4):
                raise  RuntimeError(f"Unexpected tensor size for joint_limit_minmax_pve['{joint}'], should be (2,3), but it's {limits_minmax_pve.size()}")
            std_max_pve = th.sqrt((limits_minmax_pve[0]**2+limits_minmax_pve[1]**2)/2 - (limits_minmax_pve[0]+limits_minmax_pve[1])**2/2)
            std_min_pve = th.zeros_like(limits_minmax_pve[0])
            std_minmax_pve = th.stack([std_min_pve,std_max_pve])
            ret[joint] = th.concat([limits_minmax_pve,
                                    limits_minmax_pve,
                                    limits_minmax_pve,
                                    std_minmax_pve], dim=1)
        return ret
        
    def state_names(self):
        if self._state_names is None:
            self._state_names = np.empty(shape=(self._history_length,self._fields_num)+self.field_size, dtype=object)
            for h in range(self._history_length):
                for fn in range(self._fields_num):
                    for s in np.ndindex(self.field_size):
                        jname = self.field_names[fn][1]
                        self._state_names[(h,fn)+tuple(s)] = f"[{h},{jname},{self.subfield_names[s]}]"
        return self._state_names


class JointImpedanceActionHelper:
    CONTROL_MODES = IntEnum("CONTROL_MODES", [  "VELOCITY",
                                                "TORQUE",
                                                "POSITION",
                                                "IMPEDANCE",
                                                "IMPEDANCE_NO_GAINS",
                                                "POSITION_AND_TORQUES",
                                                "POSITION_AND_STIFFNESS"], start=0)
    
    action_lengths = {
        CONTROL_MODES.IMPEDANCE: 5 ,
        CONTROL_MODES.IMPEDANCE_NO_GAINS: 3,
        CONTROL_MODES.POSITION_AND_TORQUES: 2,
        CONTROL_MODES.POSITION_AND_STIFFNESS: 2,
        CONTROL_MODES.TORQUE: 1,
        CONTROL_MODES.VELOCITY: 1,
        CONTROL_MODES.POSITION: 1,
        }
    
    def __init__(self, control_mode : CONTROL_MODES,
                        joints : Sequence[tuple[str,str]],
                        joints_minmax_pvesd : th.Tensor | dict[tuple[str,str], th.Tensor],
                        safe_stiffness : th.Tensor,
                        safe_damping : th.Tensor,
                        th_device : th.device,
                        generator : th.Generator | None):
        self._joints = joints
        self._control_mode = control_mode
        self._joints_num = len(self._joints)
        if isinstance(joints_minmax_pvesd, th.Tensor):
            self._minmax_joints_pvesd = joints_minmax_pvesd
        elif isinstance(joints_minmax_pvesd, dict):
            self._minmax_joints_pvesd = th.stack([joints_minmax_pvesd[j] for j in joints], dim=1)
        if safe_stiffness.numel() == 1:
            safe_stiffness = safe_stiffness.repeat(self._joints_num)
        if safe_damping.numel() == 1:
            safe_damping = safe_damping.repeat(self._joints_num)
        self._safe_damping = safe_damping
        self._safe_stiffness = safe_stiffness
        self._th_device = th_device

        s = normalize(self._safe_stiffness, min=self._minmax_joints_pvesd[0,:,3],max=self._minmax_joints_pvesd[1,:,3])
        d = normalize(self._safe_damping,   min=self._minmax_joints_pvesd[0,:,4],max=self._minmax_joints_pvesd[1,:,4])
        if self._control_mode == self.CONTROL_MODES.VELOCITY:
            act_to_pvesd =  [1]
            self._base_pvesd = th.as_tensor([0.0, 0.0, 0.0, -1.0, float("nan")]).repeat(self._joints_num,1)
            self._base_pvesd[:,4] = d
        elif self._control_mode == self.CONTROL_MODES.POSITION:
            act_to_pvesd =  [0]
            self._base_pvesd = th.as_tensor([0.0, 0.0, 0.0, float("nan"), float("nan")]).repeat(self._joints_num,1)
            self._base_pvesd[:,3] = s
            self._base_pvesd[:,4] = d
        elif self._control_mode == self.CONTROL_MODES.POSITION_AND_TORQUES:
            act_to_pvesd =  [0,2]
            self._base_pvesd = th.as_tensor([0.0, 0.0, 0.0, float("nan"), float("nan")]).repeat(self._joints_num,1)
            self._base_pvesd[:,3] = s
            self._base_pvesd[:,4] = d
        elif self._control_mode == self.CONTROL_MODES.IMPEDANCE_NO_GAINS:
            act_to_pvesd =  [0,1,2]
            self._base_pvesd = th.as_tensor([0.0, 0.0, 0.0, float("nan"), float("nan")]).repeat(self._joints_num,1)
            self._base_pvesd[:,3] = s
            self._base_pvesd[:,4] = d
        elif self._control_mode == self.CONTROL_MODES.IMPEDANCE:
            act_to_pvesd =  [0,1,2,3,4]
            self._base_pvesd = th.as_tensor([0.0, 0.0, 0.0, 0.0, 0.0]).repeat(self._joints_num,1)
        elif self._control_mode == self.CONTROL_MODES.POSITION_AND_STIFFNESS:
            act_to_pvesd =  [0,3]
            self._base_pvesd = th.as_tensor([0.0, 0.0, 0.0, 0.0, float("nan")]).repeat(self._joints_num,1)
            self._base_pvesd[:,4] = d
        elif self._control_mode == self.CONTROL_MODES.TORQUE:
            act_to_pvesd =  [2]
            self._base_pvesd = th.as_tensor([0.0, 0.0, 0.0, -1.0, -1.0]).repeat(self._joints_num,1)
        else:
            raise RuntimeError(f"Invalid control mode {self._control_mode}")
        self._act_to_pvesd_idx = th.as_tensor(act_to_pvesd,
                                              dtype=th.int32,
                                              device=self._th_device)
        self._action_space = spaces.ThBox(  low=-th.ones(self.action_len()),
                                            high=th.ones(self.action_len()),
                                            torch_device=self._th_device,
                                            generator=generator)
        
    def action_len(self):
        return self.action_lengths[self._control_mode]*self._joints_num
    
    def action_space(self):
        return self._action_space

    def _pvesd_to_action(self, cmds_pvesd : dict[tuple[str,str], tuple[float,float,float,float,float]]):
        cmd_joints_pvesd = th.stack([th.as_tensor(cmds_pvesd[j]) for j in self._joints])
        cmd_joints_pvesd = normalize(cmd_joints_pvesd, min=self._minmax_joints_pvesd[0], max=self._minmax_joints_pvesd[1])
        action = cmd_joints_pvesd[:,self._act_to_pvesd_idx].flatten()
        return action

    def _action_to_pvesd(self, action: th.Tensor) -> dict[tuple[str,str],tuple[float,float,float,float,float]]:
        cmd_joint_pvesd = self._base_pvesd.detach().clone()
        cmd_joint_pvesd[:,self._act_to_pvesd_idx] = action.view(self._joints_num, -1)
        cmd_joint_pvesd = unnormalize(cmd_joint_pvesd, min=self._minmax_joints_pvesd[0], max=self._minmax_joints_pvesd[1])
        if th.any(cmd_joint_pvesd[:,[3,4]] <0 ):
            ggLog.warn(f"Negative stiffness or damping!! {cmd_joint_pvesd}")
        return {self._joints[i] :  tuple(cmd_joint_pvesd[i].tolist()) for i in range(len(self._joints))}