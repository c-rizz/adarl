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
from adarl.utils.utils import masked_assign, thtens
from adarl.utils.dbg.dbg_checks import dbg_check_size, dbg_check
import numpy as np
import torch as th
import typing
from dataclasses import dataclass
import math
import numpy.typing as npt
import copy
from adarl.utils.base_utils import record_time, record_region_start, record_region_end, DelayStats


_T = TypeVar('_T', float, th.Tensor)
def unnormalize(v : _T, min : _T, max : _T, norm_min : float = -1.0, norm_max : float = 1.0) -> _T:
    return min+(v-norm_min)/(norm_max-norm_min)*(max-min)

def normalize(value : _T, min : _T, max : _T, norm_min : float = -1.0, norm_max : float = 1.0):
    return (value + (-min))/(max-min)*(norm_max-norm_min)+norm_min

def _build_full_mask(dims_masks : Sequence[th.Tensor]):
    print(f"dims_masks shapes = {[_m.shape for _m in dims_masks]}")
    dims = len(dims_masks)
    reshaped_masks = [m.view((-1,)+(1,)*(dims-i-1)) for i,m in enumerate(dims_masks)]
    print(f"reshaped_masks shapes = {[_m.shape for _m in reshaped_masks]}")
    m = reshaped_masks[0]
    for i in range(1,dims):
        print(f"Multiplying mask {i} with shape {reshaped_masks[i].shape} to current mask with shape {m.shape}")
        m = m*reshaped_masks[i]
    print(f"Resulting full mask shape = {m.shape}")
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
        full_observation_indexes : Sequence[th.Tensor]
        observed_field_size : tuple[int,...]
        observable_indexes : th.Tensor
        observable_fields : list[FieldName]
        obs_history_length : int
        obs_space : spaces.ThBox
        single_obs_space : spaces.ThBox
        fully_observable : bool
        skip_history_dim : bool = False
    @dataclass
    class SimpleObsDef():
        observable_fields : Sequence[FieldName] | None = None
        """Defines the observable fields for the observation. If None, all fields are observable."""
        observable_subfields : Sequence[str|int] | np.ndarray | None = None
        """Defines the observable subfields for the observation. If None, all subfields are observable. Only supported for 1-dimensional fields."""
        obs_history_length : int = 1
        """Defines how many history steps are observable. Must be less than or equal to the state history_length."""
        skip_history_dim : bool = False
        """Skips the history dimension in the observation. Only possible if obs_history_length==1."""

        @classmethod
        def not_observable(cls):
            return cls([],[],1)
        
        @classmethod
        def fully_observable(cls, obs_history_length : int = 1):
            return cls(obs_history_length=obs_history_length)

    def __init__(self,  fields_minmax : Mapping[FieldName,th.Tensor|Sequence[float]|Sequence[th.Tensor]],
                        dtype : th.dtype,
                        th_device : th.device,
                        field_size : list[int] | tuple[int,...],
                        vec_size : int, 
                        field_names : Sequence[FieldName] | None = None,
                        history_length : int = 1,
                        subfield_names : list[str] | np.ndarray | None = None,
                        flatten_observation = False,
                        observation_definitions : dict[str,SimpleObsDef] | SimpleObsDef | None = None,
                        normalization_range : tuple[float,float] = (-1.0, 1.0)):
        """_summary_

        Parameters
        ----------
        fields_minmax : Mapping[FieldName,th.Tensor | Sequence[float] | Sequence[th.Tensor]]
            _description_
        dtype : th.dtype
            _description_
        th_device : th.device
            _description_
        field_size : list[int] | tuple[int,...]
            _description_
        vec_size : int
            _description_
        field_names : Sequence[FieldName] | None, optional
            _description_, by default None
        history_length : int, optional
            _description_, by default 1
        subfield_names : list[str] | np.ndarray | None, optional
            _description_, by default None
        flatten_observation : bool, optional
            _description_, by default False
        observation_definitions : dict[str,SimpleObsDef] | SimpleObsDef | None, optional
            Definitions for the possible generated observations, if None, then the state is fully observable for any observation name. By default None.
            These will be used to generate observations, the proper observation will be generated when requested with the respective name
            in observe(). 
        """
        if field_names is None:
            field_names = list(fields_minmax.keys())
        self.field_names = field_names
        self.field_shape = tuple(field_size)
        self.subfield_names = self._fix_subfield_names(subfield_names)
        self._dtype = dtype
        # Target range that normalize() maps field values into. For uint8 storage it must be (0,255)
        # and normalization is skipped entirely: values are kept raw, to be scaled downstream (e.g. in the encoder).
        self._normalization_range = (float(normalization_range[0]), float(normalization_range[1]))
        if dtype == th.uint8 and self._normalization_range != (0.0, 255.0):
            raise ValueError(f"normalization_range must be (0, 255) for uint8 dtype, got {self._normalization_range}")
        self._th_device = th_device
        self._history_length = history_length
        self._flatten_observation = flatten_observation
        self._state_names = None
        self._vec_size = vec_size
        self._field_idxs = {n:field_names.index(n) for n in field_names}
        self._field_idx_cache : dict[tuple[FieldName,...] | FieldName, th.Tensor] = {}
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
            observation_definitions = {"base":observation_definitions}
        self._obs_defs = {k:self._build_obs_def(v) for k,v in observation_definitions.items()}
        self._main_obs_def = next(iter(self._obs_defs.values()))
        

    def _build_obs_def(self, obs_def : SimpleObsDef) -> ObservationDef:
        observable_fields, observable_subfields, obs_history_length = obs_def.observable_fields, obs_def.observable_subfields, obs_def.obs_history_length
        if obs_history_length>self._history_length:
            raise RuntimeError(f"obs_history_length ({obs_history_length}) must be less than state history_length ({self._history_length})")
        if obs_def.skip_history_dim and obs_history_length!=1:
            raise RuntimeError(f"remove_history_dim is only possible if obs_history_length==1, but obs_history_length={obs_history_length}")
        if observable_subfields is None:
            observable_subfields_masks = tuple()
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
                observable_subfields_masks = (observable_subfields_mask,)
                observed_field_shape : tuple[int,...] = (th.count_nonzero(observable_subfields_mask).item(),)
            elif isinstance(observable_subfields, (np.ndarray)):
                raise NotImplementedError()
        observable_fields = self.field_names if observable_fields is None else observable_fields
        observable_fields = [f for f in self.field_names if f in observable_fields] # to ensure they are ordered
        observable_indexes = th.as_tensor([self.field_names.index(n) for n in observable_fields], dtype=th.int32).to(device=self._th_device, non_blocking=self._th_device.type=="cuda")
        observable_fields_mask = th.zeros((self._fields_num,), dtype=th.bool)
        observable_fields_mask[observable_indexes] = True
        observable_hist_mask = th.zeros((self._history_length,), dtype=th.bool)
        observable_hist_mask[:obs_history_length] = True
        obs_idx_np = np.ix_(observable_hist_mask.cpu().numpy(),
                            observable_fields_mask.cpu().numpy(),
                            *[m.cpu().numpy() for m in observable_subfields_masks])
        # Keep the index tensors on the state's device: indexing a CUDA `state` (line ~417) with
        # CPU index tensors forces a synchronous host->device copy of the indices on every observe().
        full_observation_indexes=[thtens(i, device=self._th_device) for i in obs_idx_np]



        
        obs_hist_count = int(th.count_nonzero(observable_hist_mask).item())
        obs_fields_count = int(th.count_nonzero(observable_fields_mask).item())
        if obs_def.skip_history_dim:
            unflattened_obs_shape = ( self._vec_size, obs_fields_count)+observed_field_shape
        else:
            unflattened_obs_shape = ( self._vec_size, obs_hist_count, obs_fields_count)+observed_field_shape
        # print(f"obs_history_length = {obs_history_length}")
        # print(f"observable_fields = {observable_fields}")
        # print(f"observable_subfields = {observable_subfields}")
        # print(f"observed_field_shape = {observed_field_shape}")
        # print(f"observable_subfields_mask = {observable_subfields_mask}")
        # print(f"unflattened_obs_shape = {unflattened_obs_shape}")
        obs_shape = (self._vec_size,math.prod(unflattened_obs_shape[1:])) if self._flatten_observation else unflattened_obs_shape
        obs_names = self._build_obs_names(  obs_history_length,
                                            observable_fields,
                                            observed_field_shape,
                                            observable_subfields_masks,
                                            skip_history_dim=obs_def.skip_history_dim)
        # ggLog.info(f"obsnames.shape = {obs_names.shape}, obsnames = {obs_names}")
        hlmin = self._limits_minmax[0].expand(self._state_shape)
        hlmax = self._limits_minmax[1].expand(self._state_shape)
        fully_observable = observable_fields is None and obs_history_length==self._history_length and observable_subfields is None
        full_obs_def = self.ObservationDef( obs_names=obs_names,
                                            obs_shape=obs_shape,
                                            unflattened_obs_shape=unflattened_obs_shape,
                                            full_observation_indexes=full_observation_indexes,
                                            observed_field_size=observed_field_shape, 
                                            observable_indexes=observable_indexes,
                                            observable_fields=observable_fields,
                                            obs_history_length=obs_history_length, 
                                            obs_space=None,
                                            single_obs_space=None,
                                            fully_observable=fully_observable,
                                            skip_history_dim=obs_def.skip_history_dim)
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
        if len(self.field_names) == 0:
            return th.empty((2, 0) + self.field_shape, dtype=self._dtype, device=self._th_device)
        return th.stack([th.as_tensor(fields_minmax[fn], dtype=self._dtype, device=self._th_device) for fn in self.field_names]).transpose(0,1)

    def _mapping_to_tensor(self, instantaneous_state : Mapping[FieldName,th.Tensor | float | Sequence[float]]) -> th.Tensor:
        # ggLog.info(f"self._vec_size = {self._vec_size}, self.field_size = {self.field_shape}, instantaneous_state = {instantaneous_state}")
        instantaneous_state = {k:thtens(v).view(self._vec_size, *self.field_shape) for k,v in instantaneous_state.items()}
        return th.stack([instantaneous_state[k] for k in self.field_names], dim = -len(self.field_shape)-1) # stack along the field dimension

    @override
    def reset_state(self,   initial_values : th.Tensor | SupportsFloat | Mapping[FieldName,th.Tensor | float | Sequence[float]] | None= None,
                            vec_mask: th.Tensor | None = None, old_state : th.Tensor | None = None):
        if initial_values is None:
            initial_values = th.tensor(0.0)
        if isinstance(initial_values,Mapping):
            initial_values = self._mapping_to_tensor(initial_values)
            # ggLog.info(f"resetting from mapping: initial_values = {initial_values}, size = {initial_values.size()}")
        elif isinstance(initial_values,(SupportsFloat, Sequence)):
            initial_values = th.as_tensor(initial_values)
        initial_values = initial_values.expand(self._vec_size,*self._state_shape[2:]).to(device=self._th_device, dtype=self._dtype, non_blocking=self._th_device.type=="cuda")
        # dbg_check_size(initial_values, (self._state_shape[0],)+self._state_shape[2:], msg=f" Fields are {self.field_names}, subfields are {self.subfield_names}")
        new_state = initial_values.unsqueeze(1).expand(*self._state_shape).clone() # repeat along the history dimension
        # state = initial_values.repeat(self._history_length, *((1,)*len(initial_values.size())))
        assert new_state.size() == self._state_shape,    f"Unexpected resulting state size {new_state.size()}, should be {self._state_shape}."\
                                                    f" Fields are {self.field_names}, subfields are {self.subfield_names}"
        
        if vec_mask is None:
            return new_state
        else:
            dbg_check(lambda: th.logical_or(th.all(vec_mask),
                                            th.as_tensor(old_state is not None).to(vec_mask.device, non_blocking=vec_mask.device.type=="cuda")),
                      lambda: "vec_mask is not all True but old_state is None",
                      assert_msg="vec_mask is not all True but old_state is None",
                      async_assert=True)
            if old_state is None:
                return new_state
            return masked_assign(old_state, vec_mask, new_state, inplace=False)
    
    @override
    def update(self, instantaneous_state : th.Tensor | Mapping[FieldName,th.Tensor | float | Sequence[float]], state : th.Tensor, inplace = True):
        if isinstance(instantaneous_state,Mapping):
            instantaneous_state = self._mapping_to_tensor(instantaneous_state)
        # for i in range(state.size()[1]-1,0,-1):
        #     state[:,i] = state[:,i-1]
        rolled_state = state.roll(1, dims=1)
        rolled_state[:,0] = instantaneous_state.view(self._vec_size,self._fields_num,*self.field_shape)
        if inplace:
            state.copy_(rolled_state)
            return state
        else:
            return rolled_state
    
    @override
    def check_size(self, instantaneous_state : th.Tensor | Mapping[FieldName,th.Tensor] | None = None,
                         state_th : th.Tensor | None = None,
                         state_name : str = ""):
        if instantaneous_state is not None:
            if isinstance(instantaneous_state, th.Tensor):
                dbg_check_size(instantaneous_state, (self._vec_size,self._fields_num,*self.field_shape), msg=f"Unexpected size at state '{state_name}' {instantaneous_state.size()}, should be {(self._vec_size,self._fields_num,*self.field_shape)}: ")
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
    
    def flat_obs_names(self, obs_def : ObservationDef | str | None = None):
        return self.observation_names(obs_def).flatten()
    
    @override
    def flat_state_names(self):
        return self.state_names().flatten()

    @override
    def normalize(self, state : th.Tensor, alternative_limits : th.Tensor | None = None, warn_limits_violation = False):
        if self._dtype == th.uint8:
            return state # uint8 is kept raw ([0,255]); normalization is skipped and left to downstream (e.g. the encoder)
        limits = self._limits_minmax if alternative_limits is None else alternative_limits
        ret = normalize(state, limits[0], limits[1], self._normalization_range[0], self._normalization_range[1])
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
        if self._dtype == th.uint8:
            return state
        return unnormalize(state, self._limits_minmax[0], self._limits_minmax[1], self._normalization_range[0], self._normalization_range[1])
    
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
                obs = state[:,*obs_def.full_observation_indexes]
                obs = obs.view(obs_def.unflattened_obs_shape)
                # obs = th.masked_select(state, obs_def.full_observation_mask).view(obs_def.unflattened_obs_shape)
                # obs = state[:,obs_def.full_observation_mask].view(obs_def.unflattened_obs_shape)
        if obs_def.skip_history_dim:
            obs = obs[:,0]
        if self._flatten_observation:
            obs = th.flatten(obs, start_dim=1)
        return obs

    def _build_obs_names(self, obs_history_length, observable_fields, observed_field_size, observable_subfields_masks, skip_history_dim=False):
        obs_names = np.empty(shape=(obs_history_length,len(observable_fields))+observed_field_size, dtype=object)
        # print(f"observed_field_size = {observed_field_size}")
        for h in range(obs_history_length):
            for fn in range(len(observable_fields)):
                f = observable_fields[fn]
                if isinstance(f, Enum):
                    f = f.name
                indexes = list(np.ndindex(self.field_shape))
                obs_indexes = list(np.ndindex(observed_field_size))
                counter = 0
                # print(f"indexes = {indexes}")
                for s in indexes:
                    # print(f"observable_subfields_mask[{s}] = {observable_subfields_mask[s]}")
                    if observable_subfields_masks is None or len(observable_subfields_masks)==0:
                        observable = True
                    elif len(observable_subfields_masks)==1:
                        observable = observable_subfields_masks[0][s]
                    else:
                        raise NotImplementedError(f"Multiple observable_subfields_masks not supported")
                    # print(f"observable = {observable}")
                    if not observable:
                        continue
                    obs_s = obs_indexes[counter]
                    counter+=1
                    if self.subfield_names is not None:
                        sn = self.subfield_names[s]
                    else:
                        sn = ','.join([str(i) for i in s])
                    obs_names[(h,fn)+obs_s] = f"[{h},{f},{sn}]"
        if skip_history_dim:
            obs_names = obs_names[0]
        if self._flatten_observation:
            obs_names = obs_names.flatten()
        return obs_names
    
    @override
    def observation_names(self, obs_def : ObservationDef | str | None = None):
        if obs_def is None:
            obs_def = self._main_obs_def
        elif isinstance(obs_def, str):
            obs_def = self._obs_defs[obs_def]
        return obs_def.single_obs_space.labels
    
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
        """Returns an index tensor for a list of fields. This is useful for avoiding CUDA syncs,
            as indexing directly with the list would result in a sync while this does not. Also, 
            this function caches previously used index tensors for efficiency.

        Parameters
        ----------
        field_names : tuple[FieldName,...] | FieldName
            List of field names to be selected

        Returns
        -------
        th.Tensor
            The index tensor
        """
        idx = self._field_idx_cache.get(field_names, None)
        if idx is None:
            if isinstance(field_names, Sequence):
                if not isinstance(field_names, tuple):
                    field_names = tuple(field_names)
                idx = thtens([self._field_idxs[n] for n in field_names], device=self._th_device, dtype=th.int64)
            else:
                idx = thtens([field_names], device=self._th_device, dtype=th.int64)
            self._field_idx_cache[field_names] = idx
            return idx
        return idx
    
    def subfield_idx(self, subfield_names : Sequence[FieldName]):
        if self._subfield_idxs is not None:
            return thtens([self._subfield_idxs[n] for n in subfield_names], device=self._th_device)
        else:
            # Then subfield names are just the indexes
            return thtens([int(typing.cast(int, n)) for n in subfield_names], device=self._th_device)

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
        # ggLog.info(f"building noise for helper of size: {state_helper.get_vec_space().shape}")
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
            episode_mu_std = {k:v.unsqueeze(-1) if v.dim()==1 else v for k,v in episode_mu_std.items()}
            self._episode_mu_std = th.stack([episode_mu_std[k].expand(2,*self._field_size) for k in self._field_names], dim=1)

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
            self._step_std = th.stack([step_std[k].expand(self._field_size) for k in self._field_names], dim=0)
            # self._step_std = th.as_tensor([step_std[k] for k in self._field_names], dtype=self._dtype, device=self._device).reshape(self._noise_shape)

        assert self._episode_mu_std.size() == (2,)+self._noise_shape[1:], f"{self._episode_mu_std.size()} != {(2,)+self._noise_shape}"
        assert self._step_std.size() == self._noise_shape[1:], f"{self._step_std.size()} != {self._noise_shape[1:]}"

        # # ggLog.info(f"Noise generator got [{self._episode_mu_std},{self._step_std}]")
        state_limits = state_helper.get_limits()
        self._fields_scale = state_limits[1]-state_limits[0]
        # self._episode_mu_std = self._episode_mu_std*self._fields_scale.expand(2, *self._fields_scale.size())
        # self._step_std = self._step_std*self._fields_scale
        # ggLog.info(f"Noise generator unnormalized to [{self._episode_mu_std},{self._step_std}] (field_names={self._field_names[:3]}...)")


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

    def _resample_mu(self, vec_mask : th.Tensor | None = None):
        new_ep_mustd = th.stack([adarl.utils.utils.randn_from_mustd(self._episode_mu_std,
                                                                    size = self._noise_shape,
                                                                    generator=self._rng,
                                                                    squash_sigma=self._squash_sigma),
                                        self._step_std.expand(self._noise_shape)])
        if vec_mask is None:
            self._current_ep_mustd = new_ep_mustd
        else:
            perm = (1,0) + tuple(range(2, 2+len(self._noise_shape)-1))
            masked_assign(self._current_ep_mustd.permute(perm), vec_mask, new_ep_mustd.permute(perm))

    def _generate_noise(self):        
        return adarl.utils.utils.randn_from_mustd(self._current_ep_mustd,
                                                  size = self._noise_shape,
                                                  generator=self._rng,
                                                  squash_sigma = self._squash_sigma)

    def reset_state(self, vec_mask : th.Tensor | None = None, old_state : th.Tensor | None = None):
        self._resample_mu(vec_mask)
        new_noise = th.stack([self._generate_noise() for _ in range(self._history_length)], dim=1)
        if vec_mask is None:
            return new_noise
        else:
            dbg_check(lambda: th.logical_or(th.all(vec_mask),
                                            th.as_tensor(old_state is not None).to(vec_mask.device, non_blocking=vec_mask.device.type=="cuda")),
                      lambda: "vec_mask is not all True but old_state is None",
                      async_assert=True)
            if old_state is None:
                return new_noise
            return masked_assign(old_state, vec_mask, new_noise, inplace=False)
    
    def update(self, state, inplace = True):
        # for i in range(1,self._history_length):
        #     state[:,i] = state[:,i-1]
        # state[:,0] = self._generate_noise()
        # return state
    
        # for i in range(state.size()[1]-1,0,-1):
        #     state[:,i] = state[:,i-1]
        rolled_state = state.roll(1, dims=1)
        rolled_state[:,0] = self._generate_noise()
        if inplace:
            state.copy_(rolled_state)
            return state
        else:
            return rolled_state

    def normalize(self, noise):
        return noise / self._fields_scale
    
    def unnormalize(self, noise):
        return noise*self._fields_scale


# State = TypeVar("State", bound=Mapping)
class DictStateHelper(StateHelper):

    @dataclass
    class SimpleDictObsDef():
        observable_substates : list[str]
        concatenable_substates : list[str]
        noise_generators : dict[str,StateNoiseGenerator]
        concatenated_part_name : str
    @dataclass
    class DictObsDef():
        observable_substates : list[str]
        concatenable_substates : list[str]
        vec_obs_space : spaces.gym_spaces.Dict
        single_obs_space : spaces.gym_spaces.Dict
        noise_generators : dict[str,StateNoiseGenerator]
        concatenated_part_name : str
        name : str


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
            obs_definitions = {"base":obs_definitions}
        self._init_obs_defs = obs_definitions
        self._vec_size = next(iter(state_helpers.values()))._vec_size
        for k,sh in state_helpers.items():
            if sh._vec_size != self._vec_size:
                raise RuntimeError(f"Unmatched vec_size found in state_helpers. {k} has {sh._vec_size} and {next(iter(state_helpers.keys()))} has {self._vec_size}")
        self._build_state_obs_defs()

    def _build_state_obs_defs(self):
        self._obs_defs : dict[str,DictStateHelper.DictObsDef]= {}
        vec_state_subspaces : dict[str,spaces.gym.Space] = {k:s.get_vec_space() for k,s in self.sub_helpers.items()}
        single_state_subspaces : dict[str,spaces.gym.Space] = {k:s.get_single_space() for k,s in self.sub_helpers.items()}
        all_noises = set()
        self._all_noise_generators : dict[str,StateNoiseGenerator] = {}
        self._stateobs2noise_names : dict[tuple[str,str],str] = {} # Maps a (state_name,obs_def_name) to the noise state name
        all_vec_obs_subspaces   : dict[str,spaces.gym.Space] = {}
        all_single_obs_subspaces   : dict[str,spaces.gym.Space] = {}
        for obs_def_name,init_obs_def in self._init_obs_defs.items(): # For each observation definition
            for sub_state_name,noise in init_obs_def.noise_generators.items(): # For each of its noise generators
                if noise not in all_noises: # If this noise generator was not already added
                    # Define a state to keep track of the generated noise (a noise is a state, it can have a history and stuff)
                    all_noises.add(noise) 
                    noise_state_name = sub_state_name+"_noise_"+obs_def_name
                    if noise_state_name in self.sub_helpers:
                        raise RuntimeError(f"Sub state name '{noise_state_name}' clashes with noise state name. Choose a different name to avoid this.")
                    self._all_noise_generators[noise_state_name] = noise
                    vec_state_subspaces[noise_state_name] = noise.get_vec_space()
                    single_state_subspaces[noise_state_name] = noise.get_single_space()
                self._stateobs2noise_names[(sub_state_name,obs_def_name)] = noise_state_name

            concatenable_substates = init_obs_def.concatenable_substates
            concatenated_name = init_obs_def.concatenated_part_name
            nonflat_obss = [k for k in init_obs_def.observable_substates if k not in init_obs_def.concatenable_substates]
            vec_obs_subspaces    : dict[str,spaces.gym.Space] = {k:self.sub_helpers[k].get_vec_obs_space(obs_def_name)    for k in nonflat_obss}
            single_obs_subspaces : dict[str,spaces.gym.Space] = {k:self.sub_helpers[k].get_single_obs_space(obs_def_name) for k in nonflat_obss}
            if len(concatenable_substates)>0:
                concatenateded_dtype = self.sub_helpers[concatenable_substates[0]].get_single_obs_space(obs_def_name).dtype
                for subobsname in concatenable_substates:
                    if subobsname not in init_obs_def.observable_substates:
                        raise RuntimeError(f"Field {subobsname} is present in flatten_in_obs but not in observable_fields")
                    if self.sub_helpers[subobsname].get_single_obs_space(obs_def_name).dtype != concatenateded_dtype:
                        raise RuntimeError(f"All sub observations that are flattened should have the same dtype, "
                                        f"but {concatenable_substates[0]} has {concatenateded_dtype} and {subobsname} has {self.sub_helpers[subobsname].get_single_obs_space(obs_def_name).dtype}")
                single_concatenated_part_size = typing.cast(int, 
                                                sum([np.prod(self.sub_helpers[k].get_single_obs_space(obs_def_name).shape) 
                                                    for k in concatenable_substates ]))
                obs_labels = self.observation_names(obs_def=DictStateHelper.DictObsDef(observable_substates=init_obs_def.observable_substates,
                                                                                                concatenable_substates=init_obs_def.concatenable_substates,
                                                                                                noise_generators=init_obs_def.noise_generators,
                                                                                                concatenated_part_name=init_obs_def.concatenated_part_name,
                                                                                                vec_obs_space=None,
                                                                                                single_obs_space=None,
                                                                                                name=obs_def_name)
                                                                                                )[concatenated_name]
                vec_obs_subspaces[concatenated_name] = spaces.ThBox(low = -1.0, high = 1.0,
                                                                        shape=(self._vec_size, single_concatenated_part_size,),
                                                                        dtype=concatenateded_dtype,
                                                                        labels=obs_labels)            
                single_obs_subspaces[concatenated_name] = spaces.ThBox(   low = -1.0, high = 1.0,
                                                                                shape=(single_concatenated_part_size,),
                                                                                dtype=concatenateded_dtype,
                                                                                labels=obs_labels)
            obs_def = DictStateHelper.DictObsDef(observable_substates=init_obs_def.observable_substates,
                                       concatenable_substates=init_obs_def.concatenable_substates,
                                       noise_generators=init_obs_def.noise_generators,
                                       concatenated_part_name=init_obs_def.concatenated_part_name,
                                       vec_obs_space=spaces.ThDict(vec_obs_subspaces),
                                       single_obs_space=spaces.ThDict(single_obs_subspaces),
                                       name=obs_def_name)
            all_vec_obs_subspaces.update({obs_def_name+"."+k:v for k,v in vec_obs_subspaces.items()})
            all_single_obs_subspaces.update({obs_def_name+"."+k:v for k,v in single_obs_subspaces.items()})
            self._obs_defs[obs_def_name] = obs_def
        self._vec_state_space = spaces.ThDict(vec_state_subspaces)
        self._single_state_space = spaces.ThDict(single_state_subspaces)
        self._full_vec_obs_space = spaces.ThDict(all_vec_obs_subspaces)
        self._full_single_obs_space = spaces.ThDict(all_single_obs_subspaces)


        
    def add_substate(self,  state_name : str,
                            state_helper : ThBoxStateHelper,
                            obs_defs : dict[str,dict[str, bool | StateNoiseGenerator | None]]) -> DictStateHelper:
        if state_name in self.sub_helpers:
            raise RuntimeError(f"state with name '{state_name}' is already present")
        state_helpers = {state_name:state_helper}
        state_helpers.update(self.sub_helpers)
        for obs_name,init_obs_def in self._init_obs_defs.items():
            if obs_name not in obs_defs:
                continue
            obs_def = obs_defs[obs_name]
            if obs_def["observable"]:
                init_obs_def.observable_substates.append(state_name)
            if obs_def["concatenate"]:
                init_obs_def.concatenable_substates.append(state_name)
            noise = obs_def["noise"]
            if noise is not None:
                if not isinstance(noise, StateNoiseGenerator):
                    raise RuntimeError(f"Noise for state '{state_name}' in obs_def '{obs_name}' should be a StateNoiseGenerator, got {type(noise)}")
                init_obs_def.noise_generators[state_name] = noise
        
        return DictStateHelper( state_helpers=state_helpers,
                                obs_definitions=self._init_obs_defs)
        
    def reorder_substates(self, new_order : list[str]) -> DictStateHelper:
        if set(new_order) != set(self.sub_helpers.keys()):
            raise RuntimeError(f"New order {new_order} does not match existing substates {list(self.sub_helpers.keys())}")
        state_helpers = {k:self.sub_helpers[k] for k in new_order}
        obs_defs = {}
        for k in self._init_obs_defs:
            obs_def = copy.deepcopy(self._init_obs_defs[k])
            obs_def.observable_substates = [s for s in new_order if s in obs_def.observable_substates]
            obs_def.concatenable_substates = [s for s in new_order if s in obs_def.concatenable_substates]
            obs_defs[k] = obs_def
        return DictStateHelper( state_helpers=state_helpers,
                                obs_definitions=obs_defs)
    
    @override
    def reset_state(self, initial_values: Mapping[str,th.Tensor|Mapping[FieldName,th.Tensor | float | Sequence[float]]] | None = None,
                            vec_mask: th.Tensor | None = None, old_state: dict[str, th.Tensor] | None = None) -> dict[str, th.Tensor]:
        if initial_values is None:
            initial_values = {k:th.tensor(0.0) for k in self.sub_helpers.keys()}
        state = {k:self.sub_helpers[k].reset_state(v,
                                                   vec_mask=vec_mask,
                                                   old_state=old_state[k] if old_state is not None else None)
                    for k,v in initial_values.items()}
        noise_state = {k:ng.reset_state(vec_mask,
                                        old_state=old_state[k] if old_state is not None else None)
                        for k,ng in self._all_noise_generators.items()}
        # noise_state = {k+"_n":ng.reset_state() for k,ng in self.noise_generators.items()}
        state.update(noise_state)
        return state        
    
    @override
    def update(self, instantaneous_state : Mapping[str,th.Tensor | Mapping[FieldName, th.Tensor]], state : Mapping[str,th.Tensor], inplace = True):
        newstate = {}
        for k,sh in self.sub_helpers.items():
            substate = sh.update(instantaneous_state[k], state[k], inplace=inplace)
            newstate[k] = substate
        for noise_name in self._all_noise_generators:
            subnoise = self._all_noise_generators[noise_name].update(state[noise_name], inplace=inplace)
            newstate[noise_name] = subnoise
        return newstate

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

    def _observe(self, noisy_state:  Mapping[str,th.Tensor], obs_def_name : str):
        record_region_start(f"DictStateHelper._observe({obs_def_name})")
        obs_def = self._obs_defs[obs_def_name]
        nonflat_obs = {k:self.sub_helpers[k].observe(noisy_state[k], obs_def=obs_def_name) for k in  obs_def.observable_substates}
        record_time(f"DictStateHelper._observe({obs_def_name}) got subobs")
        # ggLog.info(f"non_flat_obs = {nonflat_obs}")
        concatenable_parts = []
        obs = {}
        for k,subobs in nonflat_obs.items():
            if k in obs_def.concatenable_substates:
                concatenable_parts.append(self.sub_helpers[k].flatten(subobs))
            else:
                obs[k] = subobs
        if len(concatenable_parts) > 0:
            obs[obs_def.concatenated_part_name] = th.concat(concatenable_parts, dim=1)
            # if th.any(th.abs(obs[self._flatten_part_name]) > 1.0):
            #     ggLog.warn(f"observation values exceed -1,1 normalization: nonflat_obs = {nonflat_obs},\nstate = {state}")
        record_region_end(f"DictStateHelper._observe({obs_def_name})")
        return obs

    @override
    def observe(self, state:  Mapping[str,th.Tensor], obs_def_name: None | str = None):
        record_region_start("DictStateHelper.observe")
        # ggLog.info(f"observing state {state}")
        state = self.normalize(state)
        # ggLog.info(f"normalized state = {state}")

        if obs_def_name is not None:
            record_time(f"DictStateHelper.observe: single {obs_def_name}")
            noisy_state = {k:ss+state[self._stateobs2noise_names[(k,obs_def_name)]] if (k,obs_def_name) in self._stateobs2noise_names else ss for k,ss in state.items()}
            observations = self._observe(noisy_state, obs_def_name)
        else:
            observations = {}
            for obs_def_name in self._obs_defs:
                record_time(f"DictStateHelper.observe: loop {obs_def_name}")
                noisy_state = {ssname:ss+state[self._stateobs2noise_names[(ssname,obs_def_name)]] 
                                        if (ssname,obs_def_name) in self._stateobs2noise_names else ss 
                               for ssname,ss in state.items()}
                record_time(f"DictStateHelper.observe: loop {obs_def_name} added noise")
                obs = self._observe(noisy_state, obs_def_name=obs_def_name)
                observations.update({obs_def_name+"."+subobs_name:subobs for subobs_name,subobs in obs.items()})
        record_region_end("DictStateHelper.observe")
        return observations

    def _obs_names(self, obs_def : DictObsDef):
        
        concatenated_parts_names : list[str] = []
        obs_names : dict[str, npt.NDArray[np.object_]] = {}
        for k in obs_def.observable_substates:
            if k in obs_def.concatenable_substates:
                concatenated_parts_names.extend([k+"."+str(n) for n in self.sub_helpers[k].flat_obs_names(obs_def.name)])
            else:
                obs_names[k] = self.sub_helpers[k].observation_names(obs_def.name)
        if len(concatenated_parts_names) > 0:
            # ggLog.info(f"flattened_parts_names = {flattened_parts_names}")
            obs_names[obs_def.concatenated_part_name] = np.array(concatenated_parts_names)
        return obs_names

    @override    
    def observation_names(self, obs_def_name: None | str = None, obs_def : DictObsDef | None = None):
        if obs_def_name is not None:
            obs_def = self._obs_defs[obs_def_name]
        if obs_def is not None:
            return self._obs_names(obs_def)
        else:
            all_obs_fields_names : dict[str, npt.NDArray[np.object_]] = {}
            for obs_name in self._obs_defs:
                obs_fields_names = self._obs_names(self._obs_defs[obs_name])
                all_obs_fields_names.update({obs_name+"."+subobs_name:obsfield_name for subobs_name,obsfield_name in obs_fields_names.items()})
            return all_obs_fields_names

    
    @override
    def get_vec_space(self):
        return self._vec_state_space
    
    @override
    def get_single_space(self):
        return self._single_state_space
    
    @override
    def get_vec_obs_space(self, obs_def_name: None | str = None):
        if obs_def_name is None:
            return self._full_vec_obs_space
        return self._obs_defs[obs_def_name].vec_obs_space
    
    @override
    def get_single_obs_space(self, obs_def_name: None | str = None):
        if obs_def_name is None:
            return self._full_single_obs_space
        return self._obs_defs[obs_def_name].single_obs_space
    
    def get_vec_full_obs_space(self):
        return self._full_vec_obs_space
    
    def get_single_full_obs_space(self):
        return self._full_single_obs_space

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



class JointStateHelper(ThBoxStateHelper):
    def __init__(self,  joint_limit_minmax_pveae : Mapping[tuple[str,str],np.ndarray | th.Tensor],
                        stiffness_minmax : tuple[float,float] | Mapping[tuple[str,str],np.ndarray | th.Tensor],
                        damping_minmax : tuple[float,float] | Mapping[tuple[str,str],np.ndarray | th.Tensor],
                        obs_dtype : th.dtype,
                        th_device : th.device,
                        vec_size : int,
                        history_length : int = 1,
                        observation_definitions : dict[str,ThBoxStateHelper.SimpleObsDef] | ThBoxStateHelper.SimpleObsDef | None = None):
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
                            observation_definitions=observation_definitions)

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
                        dtype : th.dtype,
                        th_device : th.device,
                        vec_size : int,
                        history_length : int = 1,
                        include_senseff_and_power = False,
                        flatten_observation = False,
                        observation_definitions : dict[str,ThBoxStateHelper.SimpleObsDef] | ThBoxStateHelper.SimpleObsDef | None = None):
        self._include_senseff_and_power = include_senseff_and_power
        joint_limit_minmax_pve = {k:th.as_tensor(v) for k,v in joint_limit_minmax_pve.items()}
        acc_minmax = {jn:th.stack([minmax_pve[0,1]-minmax_pve[1,1], minmax_pve[1,1]-minmax_pve[0,1]]).unsqueeze(1) for jn,minmax_pve in joint_limit_minmax_pve.items()}
        if include_senseff_and_power:
            subfield_names = [  "minpos","minvel","minacc","mineff","minseff","minpow",
                                "maxpos","maxvel","maxacc","maxeff","maxseff","maxpow",
                                "avgpos","avgvel","avgacc","avgeff","avgseff","avgpow",
                                "stdpos","stdvel","stdacc","stdeff","stdseff","stdpow"]
            senseff_minmax = th.as_tensor([[-10_000.0], [10_000.0]], device = th_device) # Can we have better sensed effort limits?
            pow_minmax = th.as_tensor([[-1000_000.0], [1000_000.0]], device = th_device) # Can we have better power limits?
            jlims_minmax_pvaee = {jn:th.cat([minmax_pve[:,:2], 
                                             acc_minmax[jn], 
                                             minmax_pve[:,[2]], 
                                             senseff_minmax,
                                             pow_minmax], dim=1) 
                                 for jn,minmax_pve in joint_limit_minmax_pve.items()}
        else:
            subfield_names = [  "minpos","minvel","minacc","mineff",
                                "maxpos","maxvel","maxacc","maxeff",
                                "avgpos","avgvel","avgacc","avgeff",
                                "stdpos","stdvel","stdacc","stdeff"]
            jlims_minmax_pvaee = {jn:th.cat([minmax_pve[:,:2], acc_minmax[jn], minmax_pve[:,[2]]], dim=1)
                                 for jn,minmax_pve in joint_limit_minmax_pve.items()}
        super().__init__(   field_names = list(jlims_minmax_pvaee.keys()),
                            dtype = dtype,
                            th_device = th_device,
                            field_size = (len(subfield_names),),
                            fields_minmax= self._build_fields_minmax(jlims_minmax_pvaee),
                            history_length = history_length,
                            subfield_names = subfield_names,
                            vec_size=vec_size,
                            observation_definitions=observation_definitions,
                            flatten_observation=flatten_observation)

    def _build_fields_minmax(self,  joint_limit_minmax_pvaee : Mapping[tuple[str,str],np.ndarray | th.Tensor] ) -> Mapping[FieldName,th.Tensor|Sequence[float]|Sequence[th.Tensor]]:
        ret = {}
        for joint,limits_minmax_pvaee in joint_limit_minmax_pvaee.items():
            limits_minmax_pvaee = th.as_tensor(limits_minmax_pvaee)
            expected_size = (2,6) if self._include_senseff_and_power else (2,4)
            if limits_minmax_pvaee.size() != expected_size:
                raise  RuntimeError(f"Unexpected tensor size for joint_limit_minmax_pve['{joint}'], should be {expected_size}, but it's {limits_minmax_pvaee.size()}")
            std_max_pve = th.sqrt((limits_minmax_pvaee[0]**2+limits_minmax_pvaee[1]**2)/2 - ((limits_minmax_pvaee[0]+limits_minmax_pvaee[1])/2)**2)
            std_min_pve = th.zeros_like(limits_minmax_pvaee[0])
            std_minmax_pve = th.stack([std_min_pve,std_max_pve])
            ret[joint] = th.concat([limits_minmax_pvaee,
                                    limits_minmax_pvaee,
                                    limits_minmax_pvaee,
                                    std_minmax_pve], dim=1)
        # ggLog.info(f"stats minmax = \n{ret}")
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
    """Helper class for managing actions for controlling joints via joint impedance control.
    """
    CONTROL_MODES = IntEnum("CONTROL_MODES", [  "VELOCITY",
                                                "TORQUE",
                                                "POSITION",
                                                "PVESD",
                                                "PVE",
                                                "PT",
                                                "PS",
                                                "POSITION_DELTA"], start=0)
    
    action_lengths = {
        CONTROL_MODES.PVESD: 5 ,
        CONTROL_MODES.PVE: 3,
        CONTROL_MODES.PT: 2,
        CONTROL_MODES.PS: 2,
        CONTROL_MODES.TORQUE: 1,
        CONTROL_MODES.VELOCITY: 1,
        CONTROL_MODES.POSITION: 1,
        CONTROL_MODES.POSITION_DELTA: 1
        }
    
    def __init__(self, control_mode : CONTROL_MODES,
                        joints : Sequence[tuple[str,str]],
                        joints_minmax_pvesd : th.Tensor | dict[tuple[str,str], th.Tensor],
                        safe_stiffness : th.Tensor,
                        safe_damping : th.Tensor,
                        th_device : th.device,
                        generator : th.Generator | None,
                        vec_size : int,
                        center_position : th.Tensor | dict[tuple[str,str], th.Tensor],
                        position_delta_max : th.Tensor | float | None = None):
        """

        Parameters
        ----------
        control_mode : CONTROL_MODES
            Which control mode will be exposed via actual actions (Just the reference positions? All the references? Also the gains?)
        joints : Sequence[tuple[str,str]]
            Joints to be controlled, each tuple is a joint identifier (<robot name>,<joint name>), defines the order of the 
            joints in the action definition.
        joints_minmax_pvesd : th.Tensor | dict[tuple[str,str], th.Tensor]
            Min and max limits for position, velocity, effort, stiffness and damping. Either tensor of size (joints_num, 2, 5), or dict 
            with joint names as keys and values tensors of size (2,5)
        safe_stiffness : th.Tensor
            Default stiffness to used if no stiffness is specified by the action
        safe_damping : th.Tensor
            Default stiffness to used if no stiffness is specified by the action
        th_device : th.device
            Torch device to be used
        generator : th.Generator | None
            Random number generator give to be used by the underlying spaces
        vec_size : int
            Vectorization size, i.e. how many parallel environments are going to be controlled
        center_position : th.Tensor | dict[tuple[str,str], th.Tensor]
            Center joint positions for all the joints

        """
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
        if self._control_mode == self.CONTROL_MODES.POSITION_DELTA:
            if position_delta_max is None:
                raise RuntimeError("position_delta_max should be specified when using POSITION_DELTA control mode")
            self._position_delta_max = position_delta_max
        else:
            self._position_delta_max = None

        pvesd_shape = (self._vec_size, self._joints_num, 5)
        s = normalize(self._safe_stiffness, min=self._minmax_joints_pvesd[0,:,3],max=self._minmax_joints_pvesd[1,:,3])
        d = normalize(self._safe_damping,   min=self._minmax_joints_pvesd[0,:,4],max=self._minmax_joints_pvesd[1,:,4])
        if self._control_mode == self.CONTROL_MODES.VELOCITY:
            act_to_pvesd =  [1]
            self._base_v_j_pvesd = th.as_tensor([0.0, 0.0, 0.0, -1.0, float("nan")], dtype=self._dtype, device=self._th_device).expand(pvesd_shape).clone()
            self._base_v_j_pvesd[:,:,4] = d
        elif self._control_mode == self.CONTROL_MODES.POSITION or self._control_mode == self.CONTROL_MODES.POSITION_DELTA:
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
        if isinstance(center_position, dict):
            center_position = th.as_tensor([center_position[k] for k in self._joints],
                                           device=self._th_device,
                                           dtype=self._dtype)
        if th.any(center_position < self._minmax_joints_pvesd[0,:,0]) or th.any(center_position > self._minmax_joints_pvesd[1,:,0]):
            bad_min = center_position < self._minmax_joints_pvesd[0,:,0]
            bad_max = center_position > self._minmax_joints_pvesd[1,:,0]
            raise RuntimeError(f"Center position is out of bounds of the defined min and max position limits \n"
                                f" center positions:\n"
                                f"     {center_position}\n"
                                f" minmax positions:\n"
                                f"     {self._minmax_joints_pvesd[0,:,0]}\n"
                                f"     {self._minmax_joints_pvesd[1,:,0]}\n"
                                f" Joints {[self._joints[i] for i in th.nonzero(bad_min).cpu().numpy().flatten().tolist()]} exceed minimum\n"
                                f" Joints {[self._joints[i] for i in th.nonzero(bad_max).cpu().numpy().flatten().tolist()]} exceed maximum\n"
                                f" All joints = {self._joints}")
        if self._control_mode == self.CONTROL_MODES.POSITION_DELTA:
            zero_action = th.zeros(size=(self._joints_num,), dtype=self._dtype, device=self._th_device)
        else:
            zero_cmd = th.zeros(size=(1, self._joints_num, 5), dtype=self._dtype, device=self._th_device)
            zero_cmd[:,:,0] = center_position
            zero_action = self.pvesd_to_action(zero_cmd, zero_cmd[:,:,0]).view(self.single_action_len())
        high = th.ones(self.single_action_len())
        self._single_action_space = spaces.ThBox(low  = -high,
                                                 high = high,
                                                 torch_device=th_device,
                                                 generator = generator,
                                                 default_value = zero_action)
        vec_high = th.ones(self.single_action_len()).expand(size=(self._vec_size, self.single_action_len()))
        self._vec_action_space = spaces.ThBox(  low  = -vec_high,
                                                high = vec_high,
                                                torch_device=th_device,
                                                generator = generator,
                                                default_value = zero_action)
        
    def single_action_len(self):
        return self.action_lengths[self._control_mode]*self._joints_num
    
    def get_single_action_space(self):
        return self._single_action_space
    
    def get_vec_action_space(self):
        return self._vec_action_space

    def pvesd_to_action(self, cmds_pvesd : th.Tensor, prev_posref : th.Tensor | None = None) -> th.Tensor:
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
        if self._control_mode == self.CONTROL_MODES.POSITION_DELTA:
            posref = cmd_vec_joints_pvesd[:,:,0]
            delta = posref - prev_posref #type: ignore
            dbg_check(lambda: th.logical_and(th.all(delta <= self._position_delta_max), 
                                             th.all(delta >= -self._position_delta_max)),
                                             build_msg=lambda: f"Position delta exceeds maximum! delta = {delta}, max = {self._position_delta_max}",
                                             async_assert=True)
            delta = th.clamp(delta, -self._position_delta_max, self._position_delta_max)
            return delta/self._position_delta_max
        else:
            cmd_vec_joints_pvesd = normalize(cmd_vec_joints_pvesd, min=self._minmax_joints_pvesd[0], max=self._minmax_joints_pvesd[1])
            return cmd_vec_joints_pvesd[:,:,self._act_to_pvesd_idx].flatten(start_dim=1)

    def action_to_pvesd(self, action: th.Tensor, prev_posref: th.Tensor | None = None) -> th.Tensor:
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
        action = action.view(self._vec_size, self._joints_num, self.action_lengths[self._control_mode])
        if self._control_mode == self.CONTROL_MODES.POSITION_DELTA:
            cmd_vec_joint_pvesd = unnormalize(cmd_vec_joint_pvesd, min=self._minmax_joints_pvesd[0], max=self._minmax_joints_pvesd[1])
            prev_posref = prev_posref.view(self._vec_size, self._joints_num, 1)
            position_delta_max = self._position_delta_max.view(1, self._joints_num, 1) if isinstance(self._position_delta_max, th.Tensor) else self._position_delta_max
            posref = action * position_delta_max + prev_posref #type: ignore
            cmd_vec_joint_pvesd[:,:,0] = posref.view(self._vec_size, self._joints_num)
        else:
            cmd_vec_joint_pvesd[:, :, self._act_to_pvesd_idx] = action
            cmd_vec_joint_pvesd = unnormalize(cmd_vec_joint_pvesd, min=self._minmax_joints_pvesd[0], max=self._minmax_joints_pvesd[1])
        th.clamp_(cmd_vec_joint_pvesd[:,:,0], self._minmax_joints_pvesd[0,:,0], self._minmax_joints_pvesd[1,:,0])
        dbg_check(lambda: typing.cast(bool, th.all(cmd_vec_joint_pvesd[:,:,3:5] >=0 )), build_msg=lambda: f"Negative stiffness or damping!! {cmd_vec_joint_pvesd}",
                  async_assert=True)
        return cmd_vec_joint_pvesd