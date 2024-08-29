from __future__ import annotations
import torch as th
from typing import TypeVar, Sequence, Mapping, Any, SupportsFloat
import adarl.utils.spaces as spaces
from adarl.utils.tensor_trees import TensorDict
import dataclasses
import adarl.utils.utils
import numpy as np
from abc import ABC, abstractmethod
from typing_extensions import override
import typing

_T = TypeVar('_T', float, th.Tensor)
def unnormalize(v : _T, min : _T, max : _T) -> _T:
    return min+(v+1)/2*(max-min)

def normalize(value : _T, min : _T, max : _T):
    return (value + (-min))/(max-min)*2-1


FieldName = str | int | tuple[str,str]

class StateHelper(ABC):
    @abstractmethod
    def reset_state(self, initial_values):
        ...
    @abstractmethod
    def update(self, instantaneous_state, state):
        ...
    @abstractmethod
    def flatten(self, state) -> th.Tensor:
        ...
    @abstractmethod
    def flatten_names(self) -> list[str]:
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
                        subfield_names : list[str] | None = None):
        self.field_names = field_names
        self.field_size = tuple(field_size)
        if subfield_names is not None and len(subfield_names)!=np.prod(self.field_size):
            raise RuntimeError(f"If subfield_names is provided it must have as many elements as each field, but len(subfield_names)={len(subfield_names)} and field_size is {self.field_size}")
        self.subfield_names = subfield_names
        self._obs_dtype = obs_dtype
        self._th_device = th_device
        self._history_length = history_length
        self._observable_fields = field_names if observable_fields is None else observable_fields
        
        self._fully_observable = observable_fields is None 
        self._field_idxs = {n:field_names.index(n) for n in field_names}
        if self.subfield_names is not None:
            self._subfield_idxs = {n:self.subfield_names.index(n) for n in self.subfield_names}
        else:
            self._subfield_idxs = None
        self._fields_num = len(field_names)
        self._state_size = (self._history_length, self._fields_num) + self.field_size
        self._obs_size = (self._history_length, len(self._observable_fields)) + self.field_size
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
        self._limits_minmax = th.stack([th.as_tensor(fields_minmax[fn], dtype=self._obs_dtype, device=self._th_device) for fn in field_names]).transpose(0,1)
        assert self._limits_minmax.size() == (2, self._fields_num,)+self.field_size, f"failed {self._limits_minmax.size()} == {(2, self._fields_num,)+self.field_size}"

        self._observable_indexes = th.as_tensor([self.field_names.index(n) for n in self._observable_fields])
        # print(f"observable_indexes = {self._observable_indexes}")
        lmin = self._limits_minmax[0]
        hlmin = lmin.expand(self._history_length, *lmin.size())
        lmax = self._limits_minmax[1]
        hlmax = lmax.expand(self._history_length, *lmax.size())
        self._state_space = spaces.ThBox(low=hlmin, high=hlmax, shape=self._state_size)
        self._obs_space = spaces.ThBox(low=self.observe(hlmin), high=self.observe(hlmax), shape=self._obs_size, dtype=self._obs_dtype)

    def _mapping_to_tensor(self, instantaneous_state : Mapping[FieldName,th.Tensor | float | Sequence[float]]) -> th.Tensor:
        instantaneous_state = {k:th.as_tensor(v) for k,v in instantaneous_state.items()}
        instantaneous_state = {k:v.unsqueeze(0) if v.dim()==0 else v for k,v in instantaneous_state.items()}
        return th.stack([instantaneous_state[k] for k in self.field_names])

    @override
    def reset_state(self, initial_values : th.Tensor | SupportsFloat | Mapping[FieldName,th.Tensor | float | Sequence[float]]):
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
            raise RuntimeError(f"Unexpected intial value shape {initial_values.size()}, should be {self._state_size[1:]}")
        state = initial_values.repeat(self._history_length, *((1,)*len(initial_values.size())))
        assert state.size() == self._state_size, f"Unexpected resulting state size {state.size()}, should be {self._state_size}"
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
    
    @override
    def flatten_names(self):
        subfields_num = np.prod(self.field_size)
        if self.subfield_names is None:
            subnames = ["["+(','.join([str(te) for te in t]))+"]" for t in np.ndindex(self.field_size)]
        else:
            subnames = self.subfield_names
        return [f"{self.field_names[int(i/subfields_num)]}.{subnames[i%subfields_num]}"
                    for i in range(self._fields_num*subfields_num)]
    @override
    def normalize(self, state : th.Tensor):
        return normalize(state, self._limits_minmax[0], self._limits_minmax[1])
    
    @override
    def unnormalize(self, state : th.Tensor):
        return unnormalize(state, self._limits_minmax[0], self._limits_minmax[1])
    
    @override
    def observe(self, state : th.Tensor):
        if self._fully_observable:
            return state
        else:
            return state[:,self._observable_indexes]
    
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


class StateNoiseGenerator:
    def __init__(self, state_helper : ThBoxStateHelper, generator : th.Generator, 
                        episode_mu_std : Mapping[FieldName,th.Tensor] | th.Tensor | list[float] | tuple[float],
                        step_std : Mapping[FieldName,th.Tensor] | th.Tensor | list[float] | tuple[float] | float,
                        dtype : th.dtype, device : th.device):
        self._state_helper = state_helper
        self._field_names = state_helper.field_names
        self._fields_num = len(self._field_names)
        self._field_size = state_helper.field_size
        self._rng = generator
        self._device = device
        self._dtype = dtype
        self._history_length = state_helper._history_length
        self._noise_shape = state_helper.get_space().shape[1:]
        if isinstance(episode_mu_std,(list,tuple,float)):
            episode_mu_std = th.as_tensor(episode_mu_std)
        if isinstance(step_std,(list,tuple,float)):
            step_std = th.as_tensor(step_std)
        if isinstance(episode_mu_std,th.Tensor):
            if episode_mu_std.size() == (2,):
                self._episode_mu_std = episode_mu_std.expand(*((self._fields_num,)+self._field_size+(2,))).permute(2,0,1)
            elif episode_mu_std.size() == ((2,self._fields_num)+self._field_size):
                self._episode_mu_std = episode_mu_std
            else:
                raise RuntimeError(f"Unexpected episode_mu_std size {episode_mu_std.size()}, should either be (2,) or ((2,fields_num)+field_size)")
            self._episode_mu_std = self._episode_mu_std.to(dtype=self._dtype,device=self._device)
        elif isinstance(episode_mu_std,Mapping):
            self._episode_mu_std = th.as_tensor([episode_mu_std[k] for k in self._field_names], dtype=self._dtype, device=self._device).permute(1,0)

        if isinstance(step_std,th.Tensor):
            if step_std.numel() == 1:
                self._step_std = step_std.expand(*((self._fields_num,)+self._field_size))
            elif step_std.size() == ((self._fields_num,)+self._field_size):
                self._step_std = step_std
            else:
                raise RuntimeError(f"Unexpected step_std size {step_std.size()}, should either be (1,),(,) or ((fields_num,)+field_size)")
            self._step_std = self._step_std.to(dtype=self._dtype,device=self._device)
        elif isinstance(step_std,Mapping):
            self._step_std = th.as_tensor([step_std[k] for k in self._field_names], dtype=self._dtype, device=self._device).reshape(self._noise_shape)

        assert self._episode_mu_std.size() == (2,)+self._noise_shape
        assert self._step_std.size() == self._noise_shape

        self._episode_mu_std = state_helper.unnormalize(self._episode_mu_std)
        self._step_std = state_helper.unnormalize(self._step_std)


        # At the beginning of each episode a mu is sampled
        # then at each step the noise uses this mu and the step_std
        self._current_ep_mu = adarl.utils.utils.randn_from_mustd(self._episode_mu_std, generator=self._rng)
        # At each step the noise state contains the current sampled noise
        self._state_space = spaces.ThBox(low=float("-inf"), high=float("+inf"), shape=(self._history_length,)+self._noise_shape)

    def get_space(self):
        return self._state_space

    def _resample_mu(self):
        self._current_ep_mu = adarl.utils.utils.randn_from_mustd(self._episode_mu_std, generator=self._rng)

    def _generate_noise(self):
        return adarl.utils.utils.randn_from_mustd(th.stack([self._current_ep_mu,self._step_std]), generator=self._rng)

    def reset_state(self):
        self._resample_mu()
        return th.stack([self._generate_noise() for _ in range(self._history_length)])
    
    def update(self, state):
        for i in range(1,state.size()[0]):
            state[i] = state[i-1]
        state[0] = self._generate_noise()
        return state

    def normalize(self, state):
        return self._state_helper.normalize(state)
    
    def unnormalize(self, state):
        return self._state_helper.unnormalize(state)


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
        
    @override
    def reset_state(self, initial_values: State) -> State:
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
        ret.update({k+"_n":ng.normalize(state[k]) for k,ng in self.noise_generators.items()})
        return ret
    
    @override
    def unnormalize(self, state : dict[str,th.Tensor]):
        ret = {k:sh.unnormalize(state[k]) for k,sh in self.sub_helpers.items()}
        ret.update({k+"_n":ng.unnormalize(state[k]) for k,ng in self.noise_generators.items()})
        return ret

    @override
    def observe(self, state:  dict[str,th.Tensor]):
        state = self.normalize(state)
        noisy_state = {k:ss+state[k+"_n"] if k in self.noise_generators else ss for k,ss in state.items()}
        nonflat_obs = {k:self.sub_helpers[k].observe(noisy_state[k]) for k in  self._observable_fields}
        flattened_parts = []
        obs = {}
        for k,o in nonflat_obs.items():
            if k in self._flatten_in_obs:
                flattened_parts.append(o.flatten())
            else:
                obs[k] = o
        if len(flattened_parts) > 0:
            obs[self._flatten_part_name] = th.concat(flattened_parts)
        return obs

    
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
    def flatten_names(self, include_only : list[str] | None = None):
        rets = []
        for k,sh in self.sub_helpers.items():
            if include_only is None or k in include_only:
                rets.extend(sh.flatten_names())
        return rets






# class RobotStateHelper():
    
#     def __init__(self, joints : list[str], obs_dtype : th.dtype, th_device : th.device,
#                         joint_limits_minmax_pve : dict[str,th.Tensor], control_limits_minmax_pvesd : dict[str,th.Tensor],
#                         joint_state_history_length : int = 1, control_state_history_len : int = 1):
#         joints_num = len(joints)
#         self.joint_names = joints
#         self._joint_state_history_length = joint_state_history_length
#         self._control_state_history_len = control_state_history_len
#         self._joints_num = len(joints)
#         self._obs_dtype = obs_dtype
#         self._th_device = th_device
#         self.joint_state_name = "jstate"
#         self.control_state_name = "ctrlstate"
        
#         self._limits = {self.joint_state_name : th.stack([joint_limits_minmax_pve[jn] for jn in joints]),
#                         self.control_state_name : th.stack([control_limits_minmax_pvesd[jn] for jn in joints])}

#         if self._limits[self.joint_state_name].size() != (2, joints_num,3):
#             raise RuntimeError(f"Joint limits shape does not match, should be {(2, joints_num,3)}, but it's {self._limits[self.joint_state_name].size()}")
#         if self._limits[self.control_state_name].size() != (2, joints_num,5):
#             raise RuntimeError(f"Control limits shape does not match, should be {(2, joints_num,5)}, but it's {self._limits[self.control_state_name].size()}")
        
#     def build_state(self, joint_state_pve : th.Tensor, control_state_pvesd : th.Tensor) -> dict[str,th.Tensor]:
#         # joint_state_pve     = th.zeros(size=(self._joint_state_history_length, self._joints_num, 3,),
#         #                                 dtype=self._obs_dtype, device=self._th_device)
#         # control_state_pvesd = th.zeros(size=(self._control_state_history_len, self._joints_num, 5,),
#         #                                 dtype=self._obs_dtype, device=self._th_device)
#         if joint_state_pve.size()!=(self._joints_num, 3):
#             raise RuntimeError(f"joint_state_pve should have size {(self._joints_num, 3)} but it has size {joint_state_pve.size()}")
#         if control_state_pvesd.size()!=(self._joints_num, 5):
#             raise RuntimeError(f"joint_state_pve should have size {(self._joints_num, 5)} but it has size {control_state_pvesd.size()}")
#         joint_state_pve = normalize(joint_state_pve, self._limits[self.joint_state_name][0], self._limits[self.joint_state_name][1])
#         control_state_pvesd = normalize(control_state_pvesd, self._limits[self.control_state_name][0], self._limits[self.control_state_name][1])
#         joint_state_pve = joint_state_pve.repeat(self._joint_state_history_length, 1, 1)
#         control_state_pvesd = control_state_pvesd.repeat(self._joint_state_history_length, 1, 1)
#         return {self.joint_state_name   : joint_state_pve,
#                 self.control_state_name : control_state_pvesd}

        
#     def update(self, state : dict[str,th.Tensor], joint_state_pve : th.Tensor, control_state_pvesd : th.Tensor) -> dict[str,th.Tensor]:
#         joint_state_pve = state[self.joint_state_name]
#         control_state_pvesd = state[self.control_state_name]
#         for i in range(1,joint_state_pve.size()[0]):
#             joint_state_pve[i] = joint_state_pve[i-1]
#         for i in range(1,control_state_pvesd.size()[0]):
#             control_state_pvesd[i] = control_state_pvesd[i-1]
#         joint_state_pve[0] = normalize(joint_state_pve, self._limits[self.joint_state_name][0], self._limits[self.joint_state_name][1])
#         control_state_pvesd[0] = normalize(control_state_pvesd, self._limits[self.control_state_name][0], self._limits[self.control_state_name][1])
#         return state
    
#     def unnormalize(self, state):
#         return {k:unnormalize(v, self._limits[k][0], self._limits[k][1]) for k,v in state.items()}

#     def get_space(self):
#         return spaces.gym_spaces.Dict(spaces = {self.joint_state_name : 
#                                                     spaces.ThBox(low = th.full(fill_value=-1.0, 
#                                                                             size=(self._joint_state_history_length, self._joints_num, 3), 
#                                                                             dtype=self._obs_dtype, device=self._th_device),
#                                                                  high= th.full(fill_value=1.0, 
#                                                                             size=(self._joint_state_history_length, self._joints_num, 3), 
#                                                                             dtype=self._obs_dtype, device=self._th_device)),
#                                                 self.control_state_name : 
#                                                     spaces.ThBox(low = th.full(fill_value=-1.0, 
#                                                                             size=(self._control_state_history_len, self._joints_num, 3), 
#                                                                             dtype=self._obs_dtype, device=self._th_device),
#                                                                  high= th.full(fill_value=1.0, 
#                                                                             size=(self._control_state_history_len, self._joints_num, 3), 
#                                                                             dtype=self._obs_dtype, device=self._th_device))})