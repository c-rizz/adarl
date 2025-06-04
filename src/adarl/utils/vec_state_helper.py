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
from adarl.utils.dbg.dbg_checks import dbg_check_size, dbg_check
import numpy as np
import torch as th
import typing
from dataclasses import dataclass
import math

_T = TypeVar('_T', float, th.Tensor)
def unnormalize(v : _T, min : _T, max : _T) -> _T:
    return min+(v+1)/2*(max-min)

def normalize(value : _T, min : _T, max : _T):
    return (value + (-min))/(max-min)*2-1

def _build_full_mask(dims_masks : Sequence[th.Tensor]):
    dims = len(dims_masks)
    reshaped_masks = [m.view((-1,)+(1,)*(dims-i-1)) for i,m in enumerate(dims_masks)]
    m = reshaped_masks[0]
    for i in range(1,dims):
        m = m*reshaped_masks[i]
    return m
    
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
    def get_vec_space(self):
        ...
    
    @abstractmethod
    def get_vec_obs_space(self):
        ...
    
    @abstractmethod
    def get_single_space(self):
        ...
    
    @abstractmethod
    def get_single_obs_space(self):
        ...
    
    # @abstractmethod
    # def get(self, state, field_names : Sequence[FieldName]) -> th.Tensor:
    #     ...

    @abstractmethod
    def field_idx(self, field_names : Sequence[FieldName], device : th.device) -> th.Tensor:
        ...

    @abstractmethod
    def check_size(self, instantaneous_state_th : th.Tensor | None, state_th : th.Tensor | None):
        ...


class ThBoxStateHelper(StateHelper):

    @dataclass
    class ObservationDef():
        obs_names : np.ndarray
        obs_shape : tuple[int,...]
        unflattened_obs_shape : tuple[int,...]
        full_observation_mask : th.Tensor
        observed_field_size : tuple[int,...]
        observable_indexes : th.Tensor
        observable_fields : list[FieldName]
        observable_subfields_mask : th.Tensor
        obs_history_length : int
        obs_space : spaces.ThBox
        single_obs_space : spaces.ThBox
        fully_observable : bool
    @dataclass
    class SimpleObsDef():
        observable_fields : Sequence[FieldName] | None = None
        observable_subfields : list[str|int] | np.ndarray | None = None
        obs_history_length : int = 1

    def __init__(self,  field_names : Sequence[FieldName], dtype : th.dtype,
                        th_device : th.device, fields_minmax : Mapping[FieldName,th.Tensor|Sequence[float]|Sequence[th.Tensor]],
                        field_size : list[int] | tuple[int,...],
                        vec_size : int, 
                        history_length : int = 1,
                        subfield_names : list[str] | np.ndarray | None = None,
                        flatten_observation = False,
                        observation_definitions : dict[str,SimpleObsDef] | SimpleObsDef | None = None):
        self.field_names = field_names
        self.field_shape = tuple(field_size)
        self.subfield_names = self._fix_subfield_names(subfield_names)
        self._dtype = dtype
        self._th_device = th_device
        self._history_length = history_length
        self._flatten_observation = flatten_observation
        self._state_names = None
        self._vec_size = vec_size
        self._field_idxs = {n:field_names.index(n) for n in field_names}
        self._field_idx_cache = {}
        self._subfield_idxs = {self.subfield_names[idx]:idx for idx in np.ndindex(self.subfield_names.shape)} if self.subfield_names is not None else None
        self._fields_num = len(field_names)
        self._state_shape = (self._vec_size, self._history_length, self._fields_num) + self.field_shape
        self._limits_minmax = self.build_limits(fields_minmax)
        assert self._limits_minmax.size() == (2, self._fields_num,)+self.field_shape, f"failed {self._limits_minmax.size()} == {(2, self._fields_num,)+self.field_shape}"

        hlmin = self._limits_minmax[0].expand(self._state_shape)
        hlmax = self._limits_minmax[1].expand(self._state_shape)
        self._vec_state_space = spaces.ThBox(low=hlmin, high=hlmax, shape=self._state_shape)
        self._single_state_space = spaces.ThBox(low=hlmin[0], high=hlmax[0], shape=self._state_shape[1:])

        if observation_definitions is None:
            observation_definitions = ThBoxStateHelper.SimpleObsDef(None, None, 1)
        if isinstance(observation_definitions, ThBoxStateHelper.SimpleObsDef):
            observation_definitions = {"main":observation_definitions}
        self._obs_defs = {k:self._build_obs_def(v) for k,v in observation_definitions.items()}
        self._main_obs_def = next(iter(self._obs_defs.values()))
        

    def _build_obs_def(self, obs_def : SimpleObsDef) -> ObservationDef:
        observable_fields, observable_subfields, obs_history_length = obs_def.observable_fields, obs_def.observable_subfields, obs_def.obs_history_length
        if obs_history_length>self._history_length:
            raise RuntimeError(f"obs_history_length ({obs_history_length}) must be less than state history_length ({self._history_length})")
        if observable_subfields is None:
            observable_subfields_mask = th.ones(self.field_shape,dtype=th.bool)
            observed_field_shape = self.field_shape
        else:
            if len(self.field_shape)!=1:
                raise RuntimeError(f"observable_subfields is supported only with 1-dimensional fields")
            if isinstance(observable_subfields, (list,tuple)):
                observable_subfields_mask = th.zeros(self.field_shape,dtype=th.bool)
                for s in observable_subfields:
                    if isinstance(s,str):
                        if self.subfield_names is None:
                            raise RuntimeError(f"observable_subfields contains a string, but subfield_names was not specified")
                        itemindex = np.where(self.subfield_names == s)
                        observable_subfields_mask[itemindex] = True
                    elif isinstance(s,int):
                        observable_subfields_mask[s] = True
                    else:
                        raise NotImplementedError()
            elif isinstance(observable_subfields, (np.ndarray)):
                raise NotImplementedError()
            observed_field_shape : tuple[int,...] = (th.count_nonzero(observable_subfields_mask).item(),)
        observable_fields = self.field_names if observable_fields is None else observable_fields
        observable_fields = [f for f in self.field_names if f in observable_fields] # to ensure they are ordered
        observable_indexes = th.as_tensor([self.field_names.index(n) for n in observable_fields], dtype=th.int32).to(device=self._th_device, non_blocking=self._th_device.type=="cuda")
        observable_fields_mask = th.zeros((self._fields_num,), dtype=th.bool)
        observable_fields_mask[observable_indexes] = True
        observable_hist_mask = th.zeros((self._history_length,), dtype=th.bool)
        observable_hist_mask[:obs_history_length] = True
        full_observation_mask = _build_full_mask([  observable_hist_mask, 
                                                    observable_fields_mask, 
                                                    observable_subfields_mask])
        obs_hist_count = int(th.count_nonzero(observable_hist_mask).item())
        obs_fields_count = int(th.count_nonzero(observable_fields_mask).item())
        unflattened_obs_shape = ( self._vec_size, obs_hist_count, obs_fields_count)+observed_field_shape
        # print(f"obs_history_length = {obs_history_length}")
        # print(f"observable_fields = {observable_fields}")
        # print(f"observable_subfields = {observable_subfields}")
        # print(f"observed_field_shape = {observed_field_shape}")
        # print(f"unflattened_obs_shape = {unflattened_obs_shape}")
        obs_shape = (self._vec_size,math.prod(unflattened_obs_shape[1:])) if self._flatten_observation else unflattened_obs_shape
        obs_names = self._build_obs_names(  obs_history_length,
                                            observable_fields,
                                            observed_field_shape,
                                            observable_subfields_mask)
        hlmin = self._limits_minmax[0].expand(self._state_shape)
        hlmax = self._limits_minmax[1].expand(self._state_shape)
        fully_observable = observable_fields is None and obs_history_length==self._history_length and observable_subfields is None
        full_obs_def = self.ObservationDef(obs_names, obs_shape, unflattened_obs_shape, full_observation_mask, observed_field_shape, 
                                      observable_indexes, observable_fields, observable_subfields_mask, obs_history_length, 
                                      None,None, fully_observable)
        full_obs_def.obs_space = spaces.ThBox(   low=self.observe(hlmin, full_obs_def), high=self.observe(hlmax, full_obs_def), shape=full_obs_def.obs_shape,
                                    dtype=self._dtype, labels=obs_names)
        full_obs_def.single_obs_space = spaces.ThBox(low=self.observe(hlmin, full_obs_def)[0], high=self.observe(hlmax, full_obs_def)[0], shape=full_obs_def.obs_shape[1:],
                                        dtype=self._dtype, labels=obs_names)
        return full_obs_def


    def _fix_subfield_names(self, subfield_names : list[str] | np.ndarray | None):
        if subfield_names is not None:
            if isinstance(subfield_names,(list,tuple)):
                if len(self.field_shape)!=1:
                    raise RuntimeError(f"subfield_names can be a list only if fields are 1-dimensional")
                if len(subfield_names)!=self.field_shape[0]:
                    raise RuntimeError(f"subfield_names is not of size field_size[0], len(subfield_names)={len(subfield_names)} and field_size[0] is {self.field_shape[0]}")
            elif isinstance(subfield_names, np.ndarray):
                if subfield_names.shape != self.field_shape:
                    raise RuntimeError(f"subfield_names is not of size field_size, subfield_names.shape={subfield_names.shape} and field_size is {self.field_shape}")
        if isinstance(subfield_names,(list,tuple)):
            subfield_names = np.array(subfield_names, dtype=object)
        return subfield_names




    def build_limits(self, fields_minmax : Mapping[FieldName,th.Tensor|Sequence[float]|Sequence[th.Tensor]]):
        new_minmax = {}
        for n, minmax in fields_minmax.items():
            if not isinstance(minmax, th.Tensor):
                minmax = th.as_tensor(minmax)
            minmax = minmax.squeeze()
            if minmax.size() == (2,):
                minmax = minmax.expand(self.field_shape+(2,)).permute(-1,*range(len(self.field_shape)))
            if minmax.size()!=(2,)+self.field_shape:
                raise RuntimeError(f"Field {n} has size {minmax.size()}, should be {(2,)+self.field_shape}")
            new_minmax[n]=minmax
        fields_minmax = new_minmax
        return th.stack([th.as_tensor(fields_minmax[fn], dtype=self._dtype, device=self._th_device) for fn in self.field_names]).transpose(0,1)

    def _mapping_to_tensor(self, instantaneous_state : Mapping[FieldName,th.Tensor | float | Sequence[float]]) -> th.Tensor:
        # ggLog.info(f"self._vec_size = {self._vec_size}, self.field_size = {self.field_size}, instantaneous_state = {instantaneous_state}")
        instantaneous_state = {k:th.as_tensor(v).view(self._vec_size, *self.field_shape) for k,v in instantaneous_state.items()}
        return th.stack([instantaneous_state[k] for k in self.field_names], dim = -len(self.field_shape)-1) # stack along the field dimension

    @override
    def reset_state(self, initial_values : th.Tensor | SupportsFloat | Mapping[FieldName,th.Tensor | float | Sequence[float]] | None= None):
        if initial_values is None:
            initial_values = th.tensor(0.0)
        if isinstance(initial_values,Mapping):
            initial_values = self._mapping_to_tensor(initial_values)
            # ggLog.info(f"resetting from mapping: initial_values = {initial_values}, size = {initial_values.size()}")
        elif isinstance(initial_values,(SupportsFloat, Sequence)):
            initial_values = th.as_tensor(initial_values)
        initial_values = initial_values.expand(self._vec_size,*self._state_shape[2:]).to(device=self._th_device, dtype=self._dtype, non_blocking=self._th_device.type=="cuda")
        dbg_check_size(initial_values, (self._state_shape[0],)+self._state_shape[2:], msg=f" Fields are {self.field_names}, subfields are {self.subfield_names}")
        state = initial_values.unsqueeze(1).expand(*self._state_shape).clone() # repeat along the history dimension
        # state = initial_values.repeat(self._history_length, *((1,)*len(initial_values.size())))
        assert state.size() == self._state_shape,    f"Unexpected resulting state size {state.size()}, should be {self._state_shape}."\
                                                    f" Fields are {self.field_names}, subfields are {self.subfield_names}"
        return state
    
    @override
    def update(self, instantaneous_state : th.Tensor | Mapping[FieldName,th.Tensor | float | Sequence[float]], state : th.Tensor):
        if isinstance(instantaneous_state,Mapping):
            instantaneous_state = self._mapping_to_tensor(instantaneous_state)
        for i in range(state.size()[1]-1,0,-1):
            state[:,i] = state[:,i-1]
        state[:,0] = instantaneous_state.view(self._vec_size,self._fields_num,*self.field_shape)
        return state
    
    @override
    def check_size(self, instantaneous_state : th.Tensor | Mapping[FieldName,th.Tensor] | None = None,
                         state_th : th.Tensor | None = None,
                         state_name : str = ""):
        if instantaneous_state is not None:
            if isinstance(instantaneous_state, th.Tensor):
                dbg_check_size(instantaneous_state, (self._vec_size,self._fields_num,*self.field_shape))
            else:
                for k,t in instantaneous_state.items():
                    dbg_check_size(t,
                                   (self._vec_size,) + self.field_shape,
                                   msg=f"At state '{state_name}' field '{k}' ({self.field_names[k] if isinstance(k,int) and k<len(self.field_names) else None}): ")
        if state_th is not None:
            dbg_check_size(state_th, (self._vec_size,self._history_length, self._fields_num,*self.field_shape))
        
    @override
    def flatten(self, state : th.Tensor):
        return state.flatten(start_dim=1)
    
    def flat_obs_names(self):
        return self.observation_names().flatten()
    
    @override
    def flat_state_names(self):
        return self.state_names().flatten()

    @override
    def normalize(self, state : th.Tensor, alternative_limits : th.Tensor | None = None, warn_limits_violation = False):
        limits = self._limits_minmax if alternative_limits is None else alternative_limits
        ret = normalize(state, limits[0], limits[1])
        if warn_limits_violation and th.any(th.abs(ret) > 1.1):
            ggLog.warn(f"Normalization exceeded [-1.1,1.1] range: {state} with {limits[0]} & {limits[1]} = {ret}")
        # if not th.all(th.isfinite(ret)):
        #     ggLog.info( f"Nonfinite normalized vals:\n"
        #                 f"limits:\n"
        #                 f"{limits}\n"
        #                 f"ret:\n"
        #                 f"{ret}\n")
        return ret
    
    @override
    def unnormalize(self, state : th.Tensor):
        return unnormalize(state, self._limits_minmax[0], self._limits_minmax[1])
    
    @override
    def observe(self, state : th.Tensor, obs_def : ThBoxStateHelper.ObservationDef | str | None = None):
        if isinstance(obs_def, str):
            obs_def = self._obs_defs[obs_def]
        if obs_def is None:
            obs_def = self._main_obs_def
        if obs_def.fully_observable:
            obs = state
        else:
            if obs_def.observed_field_size == self.field_shape:
                obs = state[:,:obs_def.obs_history_length,obs_def.observable_indexes]
            else:
                obs = state[:,obs_def.full_observation_mask].view(obs_def.unflattened_obs_shape)
        if self._flatten_observation:
            obs = th.flatten(obs, start_dim=1)
        return obs

    def _build_obs_names(self, obs_history_length, observable_fields, observed_field_size, observable_subfields_mask):
        obs_names = np.empty(shape=(obs_history_length,len(observable_fields))+observed_field_size, dtype=object)
        for h in range(obs_history_length):
            # print(f"observed_field_size = {observed_field_size}")
            for fn in range(len(observable_fields)):
                indexes = list(np.ndindex(observed_field_size))
                # print(f"observed_field_size = {observed_field_size}")
                # print(f"indexes = {indexes}")
                for s in indexes:
                    # print(f"observable_subfields_mask[{s}] = {observable_subfields_mask[s]}")
                    if len(observable_subfields_mask.shape)>1:
                        first_element_observable = observable_subfields_mask[s].flatten()[0]
                        # print(f"first_element_observable = {first_element_observable}")
                        # print(f"observable_subfields_mask[{s}] = {observable_subfields_mask[s]}")
                        if th.any(observable_subfields_mask[s] != first_element_observable):
                            raise RuntimeError(f"subfield not fully observable or fully not observable, not supported yet")
                        observable = first_element_observable
                    else:
                        observable = observable_subfields_mask[s]
                    if not observable:
                        continue
                    f = observable_fields[fn]
                    if isinstance(f, Enum):
                        f = f.name
                    if self.subfield_names is not None:
                        sn = self.subfield_names[s]
                    else:
                        sn = ','.join([str(i) for i in s])
                    obs_names[(h,fn)+s] = f"[{h},{f},{sn}]"
        if self._flatten_observation:
            obs_names = obs_names.flatten()
        return obs_names
    
    @override
    def observation_names(self):
        return self._main_obs_def.single_obs_space.labels
    
    def state_names(self):
        if self._state_names is None:
            self._state_names = np.empty(shape=(self._history_length,self._fields_num)+self.field_shape, dtype=object)
            for h in range(self._history_length):
                for fn in range(self._fields_num):
                    for s in np.ndindex(self.field_shape):
                        f = self.field_names[fn]
                        if isinstance(f, Enum):
                            f = f.name
                        if self.subfield_names is not None:
                            self._state_names[(h,fn)+tuple(s)] = f"[{h},{f},{self.subfield_names[s]}]"
                        else:
                            self._state_names[(h,fn)+tuple(s)] = f"[{h},{f},{','.join([str(i) for i in s])}]"
        return self._state_names
    

    @override
    def get_vec_space(self):
        return self._vec_state_space
    
    @override
    def get_vec_obs_space(self, obs_def : ThBoxStateHelper.ObservationDef | str | None = None):
        if isinstance(obs_def, str):
            obs_def = self._obs_defs[obs_def]
        if obs_def is None:
            obs_def = self._main_obs_def
        return obs_def.obs_space

    @override
    def get_single_space(self):
        return self._single_state_space
    
    @override
    def get_single_obs_space(self, obs_def : ThBoxStateHelper.ObservationDef | str | None = None):
        if isinstance(obs_def, str):
            obs_def = self._obs_defs[obs_def]
        if obs_def is None:
            obs_def = self._main_obs_def
        return obs_def.single_obs_space
    
    # @override
    # def get(self, state : th.Tensor, field_names : Sequence[FieldName] | tuple[Sequence[FieldName],Sequence[FieldName]]):
    #     if len(field_names)>0 and isinstance(field_names[0],Sequence) and isinstance(field_names[1],Sequence):
    #         return state[:self.field_idx(field_names[0], device=state.device), self.subfield_idx(field_names[1], device=state.device)]
    #     else:
    #         return state[:,self.field_idx(field_names, device=state.device)]

    @override
    def field_idx(self, field_names : tuple[FieldName,...] | FieldName):
        idx = self._field_idx_cache.get(field_names, None)
        if idx is None:
            if isinstance(field_names, Sequence):
                if not isinstance(field_names, tuple):
                    field_names = tuple(field_names)
                idx = th.as_tensor([self._field_idxs[n] for n in field_names], device=self._th_device)
            else:
                idx = th.as_tensor(field_names, device=self._th_device)
            self._field_idx_cache[field_names] = idx
            return idx
        return idx
    
    def subfield_idx(self, subfield_names : Sequence[FieldName]):
        if self._subfield_idxs is not None:
            return th.as_tensor([self._subfield_idxs[n] for n in subfield_names], device=self._th_device)
        else:
            # Then subfield names are just the indexes
            return th.as_tensor([int(typing.cast(int, n)) for n in subfield_names], device=self._th_device)

    def get_limits(self):
        """_summary_

        Returns
        -------
        th.Tensor
            Tensor of size (2, len(field_names), field_size)
        """
        return self._limits_minmax
    
    def get_flattened_limits(self):
        """_summary_

        Returns
        -------
        th.Tensor
            Tensor of size (2, len(field_names)*field_size)
        """
        return self._limits_minmax.flatten(start_dim=1)

class StateNoiseGenerator:
    def __init__(self, state_helper : ThBoxStateHelper, generator : th.Generator, 
                        episode_mu_std : Mapping[FieldName,th.Tensor] | th.Tensor | list[float] | tuple[float],
                        step_std : Mapping[FieldName,th.Tensor] | th.Tensor | list[float] | tuple[float] | float,
                        dtype : th.dtype, device : th.device,
                        squash_sigma : float = 3.0):
        ggLog.info(f"building noise for helper of size: {state_helper.get_vec_space().shape}")
        self._state_helper = state_helper
        self._field_names = state_helper.field_names
        self._fields_num = len(self._field_names)
        self._field_size = state_helper.field_shape
        self._rng = generator
        self._device = device
        self._dtype = dtype
        self._history_length = state_helper._history_length
        self._noise_shape = (state_helper._vec_size,) + state_helper.get_vec_space().shape[2:]
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

        assert self._episode_mu_std.size() == (2,)+self._noise_shape[1:], f"{self._episode_mu_std.size()} != {(2,)+self._noise_shape}"
        assert self._step_std.size() == self._noise_shape[1:]

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
        limit = squash_sigma if squash_sigma>=0 else float("+inf")
        self._vec_state_space = spaces.ThBox(low=-limit, high=limit, shape=(self._noise_shape[0], self._history_length)+self._noise_shape[1:])
        self._single_state_space = spaces.ThBox(low=-limit, high=limit, shape=(self._history_length,)+self._noise_shape[1:])

    def get_vec_space(self):
        return self._vec_state_space
    
    def get_single_space(self):
        return self._single_state_space

    def _resample_mu(self):
        self._current_ep_mustd = th.stack([adarl.utils.utils.randn_from_mustd(self._episode_mu_std,
                                                                              size = self._noise_shape,
                                                                              generator=self._rng,
                                                                              squash_sigma=self._squash_sigma),
                                           self._step_std.expand(self._noise_shape)])

    def _generate_noise(self):
        return adarl.utils.utils.randn_from_mustd(self._current_ep_mustd,
                                                  size = self._noise_shape,
                                                  generator=self._rng,
                                                  squash_sigma = self._squash_sigma)

    def reset_state(self):
        self._resample_mu()
        return th.stack([self._generate_noise() for _ in range(self._history_length)], dim=1)
    
    def update(self, state):
        for i in range(1,self._history_length):
            state[:,i] = state[:,i-1]
        state[:,0] = self._generate_noise()
        return state

    def normalize(self, noise):
        return noise / self._fields_scale
    
    def unnormalize(self, noise):
        return noise*self._fields_scale


# State = TypeVar("State", bound=Mapping)
class DictStateHelper(StateHelper):

    @dataclass
    class SimpleDictObsDef():
        observable_substates : list[str]
        flattened_subobss : list[str]
        noise_generators : dict[str,StateNoiseGenerator]
        flattened_part_name : str
    @dataclass
    class DictObsDef():
        observable_substates : list[str]
        flattened_subfields : list[str]
        vec_obs_space : spaces.gym_spaces.Dict
        single_obs_space : spaces.gym_spaces.Dict
        noise_generators : dict[str,StateNoiseGenerator]
        flattened_part_name : str


    def __init__(self,  state_helpers : dict[str, ThBoxStateHelper],
                        obs_definitions : SimpleDictObsDef | dict[str,SimpleDictObsDef] | None = None):
        # Would be nice to make this recursive (i.e. make this contain also DictStateHelpers), 
        # but it becomes a bit of a mess from the typing point of view
        self.sub_helpers = state_helpers
        if obs_definitions is None:
            obs_definitions = DictStateHelper.SimpleDictObsDef(list(state_helpers.keys()),
                                                               [],
                                                               {},
                                                               "vec")
        if isinstance(obs_definitions, DictStateHelper.SimpleDictObsDef):
            obs_definitions = {"main":obs_definitions}
        self._init_obs_defs = obs_definitions
        self._vec_size = next(iter(state_helpers.values()))._vec_size
        for k,sh in state_helpers.items():
            if sh._vec_size != self._vec_size:
                raise RuntimeError(f"Unmatched vec_size found in state_helpers. {k} has {sh._vec_size} and {next(iter(state_helpers.keys()))} has {self._vec_size}")
        self._build_state_obs_defs()

    def _build_state_obs_defs(self):
        self._obs_defs : dict[str,DictStateHelper.DictObsDef]= {}
        self._obs_defs_main_name = next(iter(self._init_obs_defs.keys()))
        vec_state_subspaces : dict[str,spaces.gym.Space] = {k:s.get_vec_space() for k,s in self.sub_helpers.items()}
        single_state_subspaces : dict[str,spaces.gym.Space] = {k:s.get_single_space() for k,s in self.sub_helpers.items()}
        all_noise_names = []
        all_noises = set()
        self._all_noise_generators : dict[str,StateNoiseGenerator] = {}
        self._state2noise_names : dict[str,str] = {}
        for obsname,init_obs_def in self._init_obs_defs.items():
            for state_name,noise in init_obs_def.noise_generators.items():
                if noise not in all_noises:
                    all_noises.add(noise)
                    noise_state_name = state_name+"_noise_"+obsname
                    self._all_noise_generators[noise_state_name] = noise
                    if noise_state_name in self.sub_helpers:
                        raise RuntimeError(f"Sub state name '{noise_state_name}' clashes with noise state name. Choose a different name to avoid this.")
                    vec_state_subspaces[noise_state_name] = noise.get_vec_space()
                    single_state_subspaces[noise_state_name] = noise.get_single_space()
                    all_noise_names.append(noise_state_name)
                self._state2noise_names[state_name] = noise_state_name

            flattened_subobss = init_obs_def.flattened_subobss
            flattened_name = init_obs_def.flattened_part_name
            nonflat_obss = [k for k in init_obs_def.observable_substates if k not in init_obs_def.flattened_subobss]
            vec_obs_subspaces    : dict[str,spaces.gym.Space] = {k:self.sub_helpers[k].get_vec_obs_space(obsname)    for k in nonflat_obss}
            single_obs_subspaces : dict[str,spaces.gym.Space] = {k:self.sub_helpers[k].get_single_obs_space(obsname) for k in nonflat_obss}
            if len(flattened_subobss)>0:
                flattened_dtype = self.sub_helpers[flattened_subobss[0]].get_single_obs_space().dtype
                for subobsname in flattened_subobss:
                    if subobsname not in init_obs_def.observable_substates:
                        raise RuntimeError(f"Field {subobsname} is present in flatten_in_obs but not in observable_fields")
                    if self.sub_helpers[subobsname].get_single_obs_space().dtype != flattened_dtype:
                        raise RuntimeError(f"All sub observations that are flattened should have the same dtype, "
                                        f"but {flattened_subobss[0]} has {flattened_dtype} and {subobsname} has {self.sub_helpers[subobsname].get_single_obs_space().dtype}")
                single_flattened_part_size = typing.cast(int, 
                                                sum([np.prod(self.sub_helpers[k].get_single_obs_space().shape) 
                                                    for k in flattened_subobss ]))
                obs_labels = adarl.utils.utils.to_string_tensor(self.observation_names(obs_def=DictStateHelper.DictObsDef(observable_substates=init_obs_def.observable_substates,
                                                                                                flattened_subfields=init_obs_def.flattened_subobss,
                                                                                                noise_generators=init_obs_def.noise_generators,
                                                                                                flattened_part_name=init_obs_def.flattened_part_name,
                                                                                                vec_obs_space=None,
                                                                                                single_obs_space=None)
                                                                                                )[flattened_name])
                vec_obs_subspaces[flattened_name] = spaces.ThBox(low = -1.0, high = 1.0,
                                                                        shape=(self._vec_size, single_flattened_part_size,),
                                                                        dtype=flattened_dtype,
                                                                        labels=obs_labels)            
                single_obs_subspaces[flattened_name] = spaces.ThBox(   low = -1.0, high = 1.0,
                                                                                shape=(single_flattened_part_size,),
                                                                                dtype=flattened_dtype,
                                                                                labels=obs_labels)
            obs_def = DictStateHelper.DictObsDef(observable_substates=init_obs_def.observable_substates,
                                       flattened_subfields=init_obs_def.flattened_subobss,
                                       noise_generators=init_obs_def.noise_generators,
                                       flattened_part_name=init_obs_def.flattened_part_name,
                                       vec_obs_space=spaces.gym_spaces.Dict(vec_obs_subspaces),
                                       single_obs_space=spaces.gym_spaces.Dict(single_obs_subspaces))
            self._obs_defs[obsname] = obs_def
        self._vec_state_space = spaces.gym_spaces.Dict(vec_state_subspaces)
        self._single_state_space = spaces.gym_spaces.Dict(single_state_subspaces)
        print(f"self._obs_defs = {self._obs_defs}")


        
    def add_substate(self,  state_name : str,
                            state_helper : ThBoxStateHelper,
                            obs_defs : dict[str,dict[str, bool | StateNoiseGenerator | None]]) -> DictStateHelper:
        if state_name in self.sub_helpers:
            raise RuntimeError(f"state with name '{state_name}' is already present")
        state_helpers = {state_name:state_helper}
        state_helpers.update(self.sub_helpers)
        for obs_name,init_obs_def in self._init_obs_defs.items():
            obs_def = obs_defs[obs_name]
            if obs_def["observable"]:
                init_obs_def.observable_substates.append(state_name)
            if obs_def["flatten"]:
                init_obs_def.flattened_subobss.append(state_name)
            noise = obs_def["noise"]
            if noise is not None:
                if not isinstance(noise, StateNoiseGenerator):
                    raise RuntimeError(f"Noise for state '{state_name}' in obs_def '{obs_name}' should be a StateNoiseGenerator, got {type(noise)}")
                init_obs_def.noise_generators[state_name] = noise
        
        return DictStateHelper( state_helpers=state_helpers,
                                obs_definitions=self._init_obs_defs)
        
    
    @override
    def reset_state(self, initial_values: Mapping[str,th.Tensor|Mapping[FieldName,th.Tensor | float | Sequence[float]]] | None = None) -> dict[str, th.Tensor]:
        if initial_values is None:
            initial_values = {k:th.tensor(0.0) for k in self.sub_helpers.keys()}
        state = {k:self.sub_helpers[k].reset_state(v) for k,v in initial_values.items()}
        noise_state = {k:ng.reset_state() for k,ng in self._all_noise_generators.items()}
        # noise_state = {k+"_n":ng.reset_state() for k,ng in self.noise_generators.items()}
        state.update(noise_state)
        return state        
    
    @override
    def update(self, instantaneous_state : Mapping[str,th.Tensor | Mapping[FieldName, th.Tensor]], state : Mapping[str,th.Tensor]):
        for k,sh in self.sub_helpers.items():
            sh.update(instantaneous_state[k], state[k])
        for noise_name in self._all_noise_generators:
            self._all_noise_generators[noise_name].update(state[noise_name])

    @override
    def normalize(self, state : Mapping[str,th.Tensor]):
        ret = {k:sh.normalize(state[k]) for k,sh in self.sub_helpers.items()}
        ret.update({k:ng.normalize(state[k]) for k,ng in self._all_noise_generators.items()})
        return ret
    
    @override
    def check_size(self, instantaneous_state : Mapping[str,th.Tensor|Mapping[str,th.Tensor]] | None = None,
                         state_th : Mapping[str,th.Tensor] | None = None,
                         ignore_missing : bool = False):
        if ignore_missing:
            helpers = instantaneous_state.keys()
        else:
            helpers = self.sub_helpers.keys()
        for k in helpers:
            sh = self.sub_helpers[k]
            sh.check_size(instantaneous_state=instantaneous_state[k] if instantaneous_state is not None else None,
                          state_th=state_th[k] if state_th is not None else None,
                          state_name=str(k))
            # ggLog.info(f"Checked {k}")

    @override
    def unnormalize(self, state : dict[str,th.Tensor]):
        ret = {k:sh.unnormalize(state[k]) for k,sh in self.sub_helpers.items()}
        ret.update({k:ng.unnormalize(state[k]) for k,ng in self._all_noise_generators.items()})
        return ret

    @override
    def observe(self, state:  Mapping[str,th.Tensor], obs_def_name: None | str = None):
        # ggLog.info(f"observing state {state}")
        state = self.normalize(state)
        # ggLog.info(f"normalized state = {state}")
        noisy_state = {k:ss+state[self._state2noise_names[k]] if k in self._state2noise_names else ss for k,ss in state.items()}

        if obs_def_name is None:
            obs_def_name = self._obs_defs_main_name
        obs_def = self._obs_defs[obs_def_name]
        nonflat_obs = {k:self.sub_helpers[k].observe(noisy_state[k], obs_def=obs_def_name) for k in  obs_def.observable_substates}
        # ggLog.info(f"non_flat_obs = {nonflat_obs}")
        flattened_parts = []
        obs = {}
        for k,subobs in nonflat_obs.items():
            if k in obs_def.flattened_subfields:
                flattened_parts.append(self.sub_helpers[k].flatten(subobs))
            else:
                obs[k] = subobs
        if len(flattened_parts) > 0:
            obs[obs_def.flattened_part_name] = th.concat(flattened_parts, dim=1)
            # if th.any(th.abs(obs[self._flatten_part_name]) > 1.0):
            #     ggLog.warn(f"observation values exceed -1,1 normalization: nonflat_obs = {nonflat_obs},\nstate = {state}")
        return obs

    @override    
    def observation_names(self, obs_def_name: None | str = None, obs_def : DictObsDef | None = None):
        if obs_def_name is None:
            obs_def_name = self._obs_defs_main_name
        if obs_def is None:
            obs_def = self._obs_defs[obs_def_name]
        flattened_parts_names = []
        obs_names = {}
        for k in obs_def.observable_substates:
            if k in obs_def.flattened_subfields:
                flattened_parts_names.extend([k+"_"+str(n) for n in self.sub_helpers[k].flat_obs_names()])
            else:
                obs_names[k] = self.sub_helpers[k].observation_names()
        if len(flattened_parts_names) > 0:
            # ggLog.info(f"flattened_parts_names = {flattened_parts_names}")
            obs_names[obs_def.flattened_part_name] = flattened_parts_names
        return obs_names

    
    @override
    def get_vec_space(self):
        return self._vec_state_space
    
    @override
    def get_vec_obs_space(self, obs_def_name: None | str = None):
        if obs_def_name is None:
            obs_def_name = self._obs_defs_main_name
        return self._obs_defs[obs_def_name].vec_obs_space
    
    @override
    def get_single_space(self):
        return self._single_state_space
    
    @override
    def get_single_obs_space(self, obs_def_name: None | str = None):
        if obs_def_name is None:
            obs_def_name = self._obs_defs_main_name
        return self._obs_defs[obs_def_name].single_obs_space
    
    # @override
    # def get(self, state : dict[str,th.Tensor],
    #         field_names : Sequence[tuple[str,Sequence[FieldName] | tuple[Sequence[FieldName],Sequence[FieldName]]]]):
    #     return [self.sub_helpers[k].get(state[k], idxs) for k,idxs in field_names]
    
    # def get_t(self, state : dict[str,th.Tensor], field_names : tuple[str,Sequence[str|int]]):
    #     ss = field_names[0]
    #     return self.sub_helpers[ss].get(state[ss], field_names=field_names[1])
    
    @override
    def field_idx(self, field_names: dict[str,list[str | int]], device: th.device):
        return {k:self.sub_helpers[k].field_idx(idxs, device=device) for k,idxs in field_names.items()}
    
    @override
    def flatten(self, state : dict[str,th.Tensor], include_only : list[str] | None = None):
        rets = []
        for k,sh in self.sub_helpers.items():
            if include_only is None or k in include_only:
                rets.append(sh.flatten(state[k]))
        return th.concat(rets, dim=1)
    
    @override
    def flat_state_names(self, include_only : list[str] | None = None):
        rets = []
        for k,sh in self.sub_helpers.items():
            if include_only is None or k in include_only:
                rets.extend([f"{k}.{sn}" for sn in sh.flat_state_names()])
        return rets



class RobotStateHelper(ThBoxStateHelper):
    def __init__(self,  joint_limit_minmax_pveae : Mapping[tuple[str,str],np.ndarray | th.Tensor],
                        stiffness_minmax : tuple[float,float] | Mapping[tuple[str,str],np.ndarray | th.Tensor],
                        damping_minmax : tuple[float,float] | Mapping[tuple[str,str],np.ndarray | th.Tensor],
                        obs_dtype : th.dtype,
                        th_device : th.device,
                        vec_size : int,
                        history_length : int = 1,
                        obs_history_length : int = 1,
                        observable_joints = None,
                        observable_subfields = None):
        subfield_names = ["pos","vel","cmdeff","acc","senseff","refpos","refvel","refeff","stiff","damp"]
        self._th_device = th_device
        super().__init__(   field_names=list(joint_limit_minmax_pveae.keys()),
                            dtype=obs_dtype,
                            th_device=th_device,
                            field_size=(len(subfield_names),),
                            fields_minmax= self._build_fields_minmax(joint_limit_minmax_pveae, stiffness_minmax, damping_minmax),
                            history_length=history_length,
                            subfield_names = subfield_names,
                            vec_size=vec_size,
                            observation_definitions=ThBoxStateHelper.SimpleObsDef(observable_fields=observable_joints,
                                                                                  observable_subfields=observable_subfields,
                                                                                  obs_history_length=obs_history_length))

    def _build_fields_minmax(self,  joint_limit_minmax_pve : Mapping[tuple[str,str],np.ndarray | th.Tensor],
                                    stiffness_minmax : tuple[float,float] | Mapping[tuple[str,str],np.ndarray | th.Tensor],
                                    damping_minmax : tuple[float,float] | Mapping[tuple[str,str],np.ndarray | th.Tensor]) -> Mapping[FieldName,th.Tensor|Sequence[float]|Sequence[th.Tensor]]:
        joint_limit_minmax_pveae = {k:th.as_tensor(l) for k,l in joint_limit_minmax_pve.items()}
        joint_limit_minmax_pveae = {jn:th.cat([
                                        lim_pve,
                                        th.as_tensor([[-5000.0], [5000]], device = lim_pve.device), # Can we have better acceleration limits?
                                        th.as_tensor([[-1000_000.0], [1000_000.0]], device = lim_pve.device) # Can we have better sensed effort limits?
                                    ], dim=-1)
                        for jn,lim_pve in joint_limit_minmax_pveae.items()}
        if isinstance(stiffness_minmax,tuple):
            stiffness_minmax = {j:th.as_tensor(stiffness_minmax, device=self._th_device) for j in joint_limit_minmax_pveae}
        if isinstance(damping_minmax,tuple):
            damping_minmax = {j:th.as_tensor(damping_minmax, device=self._th_device) for j in joint_limit_minmax_pveae}
        t_joint_limit_minmax_pveae : Mapping[Any, th.Tensor] = adarl.utils.tensor_trees.map_tensor_tree(joint_limit_minmax_pveae, lambda v: th.as_tensor(v,device = self._th_device)) #type: ignore
        t_stiffness_minmax : Mapping[Any, th.Tensor] = adarl.utils.tensor_trees.map_tensor_tree(stiffness_minmax, lambda v: th.as_tensor(v,device = self._th_device)) #type: ignore
        t_damping_minmax : Mapping[Any, th.Tensor] = adarl.utils.tensor_trees.map_tensor_tree(damping_minmax, lambda v: th.as_tensor(v,device = self._th_device)) #type: ignore
        return {joint : th.stack([  limits_minmax_pveae[:,0],
                                    limits_minmax_pveae[:,1],
                                    limits_minmax_pveae[:,2],
                                    limits_minmax_pveae[:,3],
                                    limits_minmax_pveae[:,4],
                                    limits_minmax_pveae[:,0],
                                    limits_minmax_pveae[:,1],
                                    limits_minmax_pveae[:,2],
                                    t_stiffness_minmax[joint],
                                    t_damping_minmax[joint]]).permute(1,0)
                for joint,limits_minmax_pveae in t_joint_limit_minmax_pveae.items()}
        
    def build_robot_limits(self, joint_limit_minmax_pve : Mapping[tuple[str,str],np.ndarray | th.Tensor],
                            stiffness_minmax : tuple[float,float] | Mapping[tuple[str,str],np.ndarray | th.Tensor],
                            damping_minmax : tuple[float,float] | Mapping[tuple[str,str],np.ndarray | th.Tensor]):
        return super().build_limits(fields_minmax=self._build_fields_minmax(joint_limit_minmax_pve, stiffness_minmax, damping_minmax))


    def state_names(self):
        if self._state_names is None:
            self._state_names = np.empty(shape=(self._history_length,self._fields_num)+self.field_shape, dtype=object)
            for h in range(self._history_length):
                for fn in range(self._fields_num):
                    for s in np.ndindex(self.field_shape):
                        jname = self.field_names[fn][1]
                        self._state_names[(h,fn)+tuple(s)] = f"[{h},{jname},{self.subfield_names[s]}]"
        return self._state_names

class RobotStatsStateHelper(ThBoxStateHelper):
    def __init__(self,  joint_limit_minmax_pve : Mapping[tuple[str,str],np.ndarray | th.Tensor],
                        obs_dtype : th.dtype,
                        th_device : th.device,
                        vec_size : int,
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
                            dtype = obs_dtype,
                            th_device = th_device,
                            field_size = (len(subfield_names),),
                            fields_minmax= self._build_fields_minmax(jlims_minmax_pvae),
                            history_length = history_length,
                            subfield_names = subfield_names,
                            vec_size=vec_size)

    def _build_fields_minmax(self,  joint_limit_minmax_pve : Mapping[tuple[str,str],np.ndarray | th.Tensor]
                             ) -> Mapping[FieldName,th.Tensor|Sequence[float]|Sequence[th.Tensor]]:
        ret = {}
        for joint,limits_minmax_pve in joint_limit_minmax_pve.items():
            limits_minmax_pve = th.as_tensor(limits_minmax_pve)
            if limits_minmax_pve.size() != (2,4):
                raise  RuntimeError(f"Unexpected tensor size for joint_limit_minmax_pve['{joint}'], should be (2,3), but it's {limits_minmax_pve.size()}")
            std_max_pve = th.sqrt((limits_minmax_pve[0]**2+limits_minmax_pve[1]**2)/2 - ((limits_minmax_pve[0]+limits_minmax_pve[1])/2)**2)
            std_min_pve = th.zeros_like(limits_minmax_pve[0])
            std_minmax_pve = th.stack([std_min_pve,std_max_pve])
            ret[joint] = th.concat([limits_minmax_pve,
                                    limits_minmax_pve,
                                    limits_minmax_pve,
                                    std_minmax_pve], dim=1)
        ggLog.info(f"stats minmax = \n{ret}")
        return ret
        
    def state_names(self):
        if self._state_names is None:
            self._state_names = np.empty(shape=(self._history_length,self._fields_num)+self.field_shape, dtype=object)
            for h in range(self._history_length):
                for fn in range(self._fields_num):
                    for s in np.ndindex(self.field_shape):
                        jname = self.field_names[fn][1]
                        self._state_names[(h,fn)+tuple(s)] = f"[{h},{jname},{self.subfield_names[s]}]"
        return self._state_names


class JointImpedanceActionHelper:
    CONTROL_MODES = IntEnum("CONTROL_MODES", [  "VELOCITY",
                                                "TORQUE",
                                                "POSITION",
                                                "PVESD",
                                                "PVE",
                                                "PT",
                                                "PS"], start=0)
    
    action_lengths = {
        CONTROL_MODES.PVESD: 5 ,
        CONTROL_MODES.PVE: 3,
        CONTROL_MODES.PT: 2,
        CONTROL_MODES.PS: 2,
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
                        generator : th.Generator | None,
                        vec_size : int = 1):
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
        self._vec_size = vec_size
        self._dtype = th.float32

        pvesd_shape = (self._vec_size, self._joints_num, 5)
        s = normalize(self._safe_stiffness, min=self._minmax_joints_pvesd[0,:,3],max=self._minmax_joints_pvesd[1,:,3])
        d = normalize(self._safe_damping,   min=self._minmax_joints_pvesd[0,:,4],max=self._minmax_joints_pvesd[1,:,4])
        if self._control_mode == self.CONTROL_MODES.VELOCITY:
            act_to_pvesd =  [1]
            self._base_v_j_pvesd = th.as_tensor([0.0, 0.0, 0.0, -1.0, float("nan")], dtype=self._dtype, device=self._th_device).expand(pvesd_shape).clone()
            self._base_v_j_pvesd[:,:,4] = d
        elif self._control_mode == self.CONTROL_MODES.POSITION:
            act_to_pvesd =  [0]
            self._base_v_j_pvesd = th.as_tensor([0.0, 0.0, 0.0, float("nan"), float("nan")], dtype=self._dtype, device=self._th_device).expand(pvesd_shape).clone()
            self._base_v_j_pvesd[:,:,3] = s
            self._base_v_j_pvesd[:,:,4] = d
        elif self._control_mode == self.CONTROL_MODES.PT:
            act_to_pvesd =  [0,2]
            self._base_v_j_pvesd = th.as_tensor([0.0, 0.0, 0.0, float("nan"), float("nan")], dtype=self._dtype, device=self._th_device).expand(pvesd_shape).clone()
            self._base_v_j_pvesd[:,:,3] = s
            self._base_v_j_pvesd[:,:,4] = d
        elif self._control_mode == self.CONTROL_MODES.PVE:
            act_to_pvesd =  [0,1,2]
            self._base_v_j_pvesd = th.as_tensor([0.0, 0.0, 0.0, float("nan"), float("nan")], dtype=self._dtype, device=self._th_device).expand(pvesd_shape).clone()
            self._base_v_j_pvesd[:,:,3] = s
            self._base_v_j_pvesd[:,:,4] = d
        elif self._control_mode == self.CONTROL_MODES.PVESD:
            act_to_pvesd =  [0,1,2,3,4]
            self._base_v_j_pvesd = th.as_tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=self._dtype, device=self._th_device).expand(pvesd_shape).clone()
        elif self._control_mode == self.CONTROL_MODES.PS:
            act_to_pvesd =  [0,3]
            self._base_v_j_pvesd = th.as_tensor([0.0, 0.0, 0.0, 0.0, float("nan")], dtype=self._dtype, device=self._th_device).expand(pvesd_shape).clone()
            self._base_v_j_pvesd[:,:,4] = d
        elif self._control_mode == self.CONTROL_MODES.TORQUE:
            act_to_pvesd =  [2]
            self._base_v_j_pvesd = th.as_tensor([0.0, 0.0, 0.0, -1.0, -1.0], dtype=self._dtype, device=self._th_device).expand(pvesd_shape).clone()
        else:
            raise RuntimeError(f"Invalid control mode {self._control_mode}")
        self._act_to_pvesd_idx = th.as_tensor(act_to_pvesd,
                                              dtype=th.int32,
                                              device=self._th_device)
        high = th.ones(self.single_action_len())
        self._single_action_space = spaces.ThBox(low  = -high,
                                                 high = high,
                                                 torch_device=th_device,
                                                 generator = generator)
        vec_high = th.ones(self.single_action_len()).expand(size=(self._vec_size, self.single_action_len()))
        self._vec_action_space = spaces.ThBox(  low  = -vec_high,
                                                high = vec_high,
                                                torch_device=th_device,
                                                generator = generator)
        
    def single_action_len(self):
        return self.action_lengths[self._control_mode]*self._joints_num
    
    def get_single_action_space(self):
        return self._single_action_space
    
    def get_vec_action_space(self):
        return self._vec_action_space

    def pvesd_to_action(self, cmds_pvesd : th.Tensor) -> th.Tensor:
        """Converts a joint impedance command (pvesd) to its respective action.

        Parameters
        ----------
        cmds_pvesd : th.Tensor
            Tensor of size (vec_size, len(joints), 5). The last dimension contains
            the position, velocity, effort, stiffness, damping (pvesd) command. The order of the joints is the one used
            in the joints argument of the constructor.

        Returns
        -------
        th.Tensor
            Tensor of size (vec_size, len(joints) *  action_len()).
        """
        if isinstance(cmds_pvesd, th.Tensor):
            cmd_vec_joints_pvesd = cmds_pvesd
        else:
            cmd_vec_joints_pvesd = th.stack([th.as_tensor(cmds_pvesd[j], device=self._th_device) for j in self._joints]).unsqueeze(0).expand(self._vec_size, len(self._joints), 5)
        cmd_vec_joints_pvesd = normalize(cmd_vec_joints_pvesd, min=self._minmax_joints_pvesd[0], max=self._minmax_joints_pvesd[1])
        return cmd_vec_joints_pvesd[:,:,self._act_to_pvesd_idx].flatten(start_dim=1)

    def action_to_pvesd(self, action: th.Tensor) -> th.Tensor:
        """Converts an action to its respective joint impedance command (pvesd)

        Parameters
        ----------
        action : th.Tensor
            Tensor of size (vec_size, len(joints) *  action_len())

        Returns
        -------
        th.Tensor
            Tensor of size (vec_size, len(joints), 5)
        """

        cmd_vec_joint_pvesd = self._base_v_j_pvesd.detach().clone()
        cmd_vec_joint_pvesd[:, :, self._act_to_pvesd_idx] = action.view(self._vec_size, self._joints_num, self.action_lengths[self._control_mode])
        cmd_vec_joint_pvesd = unnormalize(cmd_vec_joint_pvesd, min=self._minmax_joints_pvesd[0], max=self._minmax_joints_pvesd[1])
        dbg_check(lambda: typing.cast(bool, th.all(cmd_vec_joint_pvesd[:,:,[3,4]] >=0 )), build_msg=lambda: f"Negative stiffness or damping!! {cmd_vec_joint_pvesd}")
        return cmd_vec_joint_pvesd