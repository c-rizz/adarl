from dataclasses import dataclass
import torch as th
from enum import IntEnum, Enum, EnumMeta
from collections import namedtuple
from typing import TypeVar, Generic, Type, Iterable, Any

FieldId = TypeVar("FieldId", bound=IntEnum) # Must be a class derived from IntEnum
@dataclass
class TensorStruct(Generic[FieldId]):

    def __init__(self, device : th.device, dtype : th.dtype, fields : Type[FieldId]):
        self.fields = fields
        self._device = device
        self._dtype = dtype
        self._enum_start = min([f.value for f in fields])
        self.tensor = th.empty(size= (len(fields),), device=device, dtype=dtype)
        
    def fill(self, indexes : th.Tensor, values : th.Tensor):
        self.tensor[indexes] = values

    def set_raw(self, **elements):
        """Allows to set the elements like:
            tensorstruct.set_raw(a=1.0, b=2, c=3)
        """
        names, values = zip(*elements.items())
        indexer = self.get_indexer([getattr(self.fields,name) for name in names])
        self.tensor[indexer] = th.as_tensor(values, device=self._device)

    def set_dict(self, elements : dict[FieldId, Any]):
        """Allows to set the elements like:
            tensorstruct.set_dict({ts.fields.a:1.0, ts.fields.b:2, ts.fields.c:3})
        """
        field_ids, values = zip(*elements.items())
        indexer = self.get_indexer(field_ids)
        self.tensor[indexer] = th.as_tensor(values, device=self._device)

    def get_indexer(self, fields : Iterable[FieldId]):
        return th.tensor([self.get_index(n) for n in fields], device=self._device)
    
    def get_index(self, field : FieldId):
        return field.value-self._enum_start

    def get(self, indexes : th.Tensor):
        return self.tensor[indexes]
    
    def __getattr__(self, name):
        return self.tensor[th.tensor(self.get_index(getattr(self.fields,name)))]
    
    def __setattr__(self, name, value):
        if name not in ["fields","_device","_dtype","_enum_start","_tensor"] and hasattr(self.fields, name):
            self.tensor[th.tensor(self.get_index(getattr(self.fields,name)))] = value
        else:
            super().__setattr__(name, value)
    
    def __getitem__(self, key):
        if isinstance(key, th.Tensor):
            pass
        elif isinstance(key, Iterable):
            key = self.get_indexer(key)
        else:
            key = th.tensor(self.get_index(key))
        return self.get(key)
    
    def __setitem__(self, key, value):
        if isinstance(key, th.Tensor):
            pass
        elif isinstance(key, Iterable):
            key = self.get_indexer(key)
        else:
            key = th.tensor(self.get_index(key))
        self.tensor[key] = value


if __name__=="__main__":
    ts = TensorStruct(device=th.device("cuda"), dtype=th.float32,
                      fields=IntEnum("Fields",['a','b','c']))    
    print(f"ts.fields.a = {ts.fields.a}")
    ts.fill(ts.get_indexer([ts.fields.a]),th.tensor([42.0]))
    print(f"ts[ts.fields.a] = {ts[ts.fields.a]}")
    print(f"ts.a = {ts.a}")
    ts[ts.fields.c] = 44
    ts.b = 15.0
    ac = ts.get_indexer([ts.fields.a, ts.fields.c])
    print(f"ts[ac] = {ts[ac]}")
    print(f"ts['b'] = {ts[ts.fields.b]}")
    print(f"ts = {ts[[f for f in ts.fields]]}")

    ts.set_dict({ts.fields.a : 3.0,
            ts.fields.c : 5,
            ts.fields.b : 4})
    
    print(f"ts = {ts[[f for f in ts.fields]]}")

    
    