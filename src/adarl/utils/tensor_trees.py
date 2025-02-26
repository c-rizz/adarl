from __future__ import annotations
from adarl.utils.spaces import gym_spaces, ThBox
import torch as th
from typing import Optional, Any, List, Dict, Tuple, Union, Callable, TypeVar, Mapping, Sequence
import numpy as np
from adarl.utils.utils import torch_to_numpy_dtype_dict
import dataclasses
import adarl.utils.dbg.ggLog as ggLog

LeafType = TypeVar("LeafType")
TensorMapping = Union[Mapping[Any,"TensorMapping[LeafType]"], LeafType]
TensorTree = Union[Mapping[Any,"TensorTree[LeafType]"],
                   Sequence["TensorTree[LeafType]"],
                   Tuple["TensorTree[LeafType]", ...],
                   
                   LeafType]

def create_tensor_tree(batch_size : int, space : gym_spaces.Space, share_mem : bool, device : th.device) -> TensorTree[th.Tensor]:
    if isinstance(space, gym_spaces.Dict):
        tree = {}
        for k, s in space.spaces.items():
            tree[k] = create_tensor_tree(batch_size, s, share_mem, device)
        return tree
    elif isinstance(space, gym_spaces.Box):
        thdtype = th.as_tensor(space.high).dtype
        t = th.zeros(size=(batch_size,)+space.shape, dtype=thdtype, device = device)
        if share_mem:
            t.share_memory_()
        return t
    else:
        raise RuntimeError(f"Unsupported space {space}")

def fill_tensor_tree(env_idx : Optional[int], src_tree : TensorTree, dst_tree : TensorTree, depth = 0, nonstrict=False, non_blocking = False):
    if isinstance(src_tree, dict):
        if not isinstance(dst_tree,dict):
            raise RuntimeError(f"Tree element type mismatch. src = {type(src_tree)}, dst = {dst_tree}")
        if nonstrict:
            src_keys = set(src_tree.keys())
            dst_keys = set(dst_tree.keys())
            fill_keys = src_keys.intersection(dst_keys)
        elif src_tree.keys() != dst_tree.keys():
            src_keys = set(src_tree.keys())
            dst_keys = set(dst_tree.keys())
            raise RuntimeError(f"source and destination keys do not match:\nsrc={src_keys}\ndst={dst_keys}\ndiff={(src_keys-dst_keys).union(dst_keys-src_keys)}")
        else:
            fill_keys = dst_tree.keys()
        for k in fill_keys:
            fill_tensor_tree(env_idx, src_tree[k], dst_tree[k], depth = depth+1, nonstrict=nonstrict,non_blocking=non_blocking)
    elif isinstance(src_tree, th.Tensor):
        if not isinstance(dst_tree,th.Tensor):
            raise RuntimeError(f"Tree element type mismatch. src = {type(src_tree)}, dst = {dst_tree}")
        if env_idx is not None:
            dst_tree[env_idx].copy_(src_tree, non_blocking=non_blocking)
        else:
            dst_tree.copy_(src_tree, non_blocking=non_blocking)
    else:
        raise RuntimeError(f"Unexpected tree element type {type(src_tree)}")

T = TypeVar('T')
U = TypeVar('U')
def map_tensor_tree(src_tree : TensorTree[U], func : Callable[[U],T], _key = "") -> TensorTree[T]:
    if isinstance(src_tree, dict):
        r = {}
        for k in src_tree.keys():
            r[k] = map_tensor_tree(src_tree[k], func = func, _key = _key+f".{k}")
        return r
    elif isinstance(src_tree, tuple):
        r = tuple([map_tensor_tree(e, func = func, _key = _key+f".T{i}") for i,e in enumerate(src_tree)])
        return r
    elif isinstance(src_tree, list):
        return [map_tensor_tree(e, func = func, _key = _key+f".L{i}") for i,e in enumerate(src_tree)]
    elif dataclasses.is_dataclass(src_tree):
        mapped_fields = {field.name: map_tensor_tree(getattr(src_tree, field.name), func = func, _key = _key+f"{field.name}")
                          for field in dataclasses.fields(src_tree)}
        return src_tree.__class__(**mapped_fields)
    else:
        try:
            return func(src_tree)
        except Exception as e:
            raise RuntimeError(f"Exception at element {_key}: {e}")


_discarded = object()
def filter_tensor_tree(src_tree : TensorTree[U], keep : Callable[[U],bool], _already_filtered : dict = {}) -> TensorTree[U]:
    if isinstance(src_tree, dict):
        if id(src_tree) in _already_filtered:
            return _already_filtered[id(src_tree)]
        filtered = {}
        _already_filtered[id(src_tree)] = filtered
        _filtered = {k:filter_tensor_tree(v, keep=keep) for k,v in src_tree.items()}
        _filtered = {k:v for k,v in filtered.items() if v is not _discarded}
        filtered.update(_filtered)
        return filtered
    elif isinstance(src_tree, tuple):
        if id(src_tree) in _already_filtered:
            return tuple(_already_filtered[id(src_tree)])
        filtered = []
        _already_filtered[id(src_tree)] = filtered
        _filtered = [filter_tensor_tree(v, keep=keep) for v in src_tree]
        _filtered = [v for v in filtered if v is not _discarded]
        filtered.extend(filtered)
        return tuple(filtered)
    elif isinstance(src_tree, list):
        if id(src_tree) in _already_filtered:
            return _already_filtered[id(src_tree)]
        filtered = []
        _already_filtered[id(src_tree)] = filtered
        _filtered = [filter_tensor_tree(v, keep=keep) for v in src_tree]
        _filtered = [v for v in filtered if v is not _discarded]
        filtered.extend(_filtered)
        return filtered
    elif dataclasses.is_dataclass(src_tree):
        raise RuntimeError(f"Cannot filter dataclasses")
    else:
        return src_tree if keep(src_tree) else _discarded

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')
def map2_tensor_tree(src_tree1 : TensorTree[U],
                     src_tree2 : TensorTree[V],
                     func : Callable[[U,V],T]) -> TensorTree[T]:
    if isinstance(src_tree1, dict):
        if not isinstance(src_tree2, dict):
            raise RuntimeError(f"Tensor tree types do not match {type(src_tree1)} != {type(src_tree2)}")
        r = {}
        for k in src_tree1.keys():
            r[k] = map2_tensor_tree(src_tree1[k], src_tree2[k], func = func)
        return r
    elif isinstance(src_tree1, tuple):
        if not isinstance(src_tree2, tuple):
            raise RuntimeError(f"Tensor tree types do not match {type(src_tree1)} != {type(src_tree2)}")
        r : list[TensorTree[T]] = [None]*len(src_tree1) # type: ignore
        for k in range(len(src_tree1)):
            r[k] = map2_tensor_tree(src_tree1[k], src_tree2[k], func = func)
        return tuple(r)    
    elif isinstance(src_tree1, list):
        if not isinstance(src_tree2, list):
            raise RuntimeError(f"Tensor tree types do not match {type(src_tree1)} != {type(src_tree2)}")
        r : list[TensorTree[T]] = [None]*len(src_tree1) # type: ignore
        for k in range(len(src_tree1)):
            r[k] = map2_tensor_tree(src_tree1[k], src_tree2[k], func = func)
        return r
    elif dataclasses.is_dataclass(src_tree1):
        if dataclasses.is_dataclass(src_tree2):
            raise RuntimeError(f"Tensor tree types do not match {type(src_tree1)} != {type(src_tree2)}")
        mapped_fields = {}
        for field in dataclasses.fields(src_tree1):
            mapped_fields[field.name] = map2_tensor_tree(getattr(src_tree1, field.name), getattr(src_tree2, field.name), func = func)
        return src_tree1.__class__(**mapped_fields)
    else:
        # if not isinstance(src_tree2, type(src_tree1)):
        #     raise RuntimeError(f"Tensor tree types do not match {type(src_tree1)} != {type(src_tree2)}")
        return func(src_tree1, src_tree2)


T = TypeVar('T')
def flatten_tensor_tree(src_tree : TensorTree[T]) -> dict[tuple,T]:
    """Flattens a tensor tree, returning a tensor tree with not subtrees, 
    defined as a dictionary, with tuples as keys.

    Parameters
    ----------
    src_tree : TensorTree
        Tensor tree to ble flattened

    Returns
    -------
    dict
        _description_
    """
    if isinstance(src_tree, tuple):
        src_tree = {f"T{i}":src_tree[i] for i in range(len(src_tree))}
    elif isinstance(src_tree, list):
        src_tree = {f"L{i}":src_tree[i] for i in range(len(src_tree))}
    elif dataclasses.is_dataclass(src_tree):
        src_tree = {f"D{k}":v for k,v in dataclasses.asdict(src_tree).items()}
    if isinstance(src_tree, dict):
        r = {}
        for k in src_tree.keys():
            flat_subtree = flatten_tensor_tree(src_tree[k])
            for sk,sv in flat_subtree.items():
                r[(k,)+sk] = sv
        return r
    else:
        return {tuple():src_tree}

def stack_tensor_tree(src_trees : List[TensorTree]) -> TensorTree:
    if isinstance(src_trees[0], th.Tensor):
        # assume all trees are matching, thus they are all tensors
        return th.stack(src_trees)
    elif isinstance(src_trees[0], dict):
        d = {}
        for k in src_trees[0].keys():
            d[k] = stack_tensor_tree([src_trees[i][k] for i in range(len(src_trees))])
        return d
    elif isinstance(src_trees[0], list):
        l = []
        for i in range(len(src_trees[0])):
            l.append(stack_tensor_tree([src_trees[i][j] for j in range(len(src_trees))]))
        return l
    elif isinstance(src_trees[0], tuple):
        l = []
        for i in range(len(src_trees[0])):
            l.append(stack_tensor_tree([src_trees[i][j] for j in range(len(src_trees))]))
        return tuple(l)
    else:
        raise RuntimeError(f"Unexpected tensor tree type {type(src_trees[0])}")

def index_select_tensor_tree(src_tree: TensorTree, indexes : th.Tensor) -> TensorTree:
    return map_tensor_tree(src_tree=src_tree, func=lambda t: t.index_select(dim = 0, index=indexes))




def unstack_tensor_tree(src_tree : TensorTree, _key = "", _already_unstacked : dict = {}) -> list:
    if isinstance(src_tree, dict):
        # print(f"src_tree = {src_tree}")
        dictlist = []
        if id(src_tree) in _already_unstacked:
            raise RuntimeError(f"Cycle in tensor tree at {_key}")
            dictlist = _already_unstacked[id(src_tree)]
        for k in src_tree.keys():
            unstacked_subtree = unstack_tensor_tree(src_tree[k], _key=_key+f".{k}", _already_unstacked=_already_unstacked)
            stack_size = len(unstacked_subtree)
            if len(dictlist)==0:
                dictlist.extend([{} for _ in range(stack_size)])
            for i in range(stack_size):
                try:
                    dictlist[i][k] = unstacked_subtree[i]
                except Exception as e:
                    raise RuntimeError(f"exception at {_key+'.'+k if _key != '' else k}[{i}]: {e}")
        # print(f"src_tree = {src_tree}, unstacked = {dictlist}")
        return dictlist
    elif isinstance(src_tree, th.Tensor):
        ret = src_tree.unbind()
        # print(f"src_tree = {src_tree}, unstacked = {ret}")
        return ret
    else:
        raise RuntimeError(f"Unexpected tree element type {type(src_tree)} at key {_key}")

def space_from_tree(tensor_tree, labels = None):
    if isinstance(tensor_tree, dict):
        subspaces = {}
        for k in tensor_tree.keys():
            try:
                subspaces[k] = space_from_tree(tensor_tree[k], labels.get(k,None) if labels is not None else None)
            except RuntimeError as e:
                raise RuntimeError(f"{k}.{e}")    
        return gym_spaces.Dict(subspaces)
    if isinstance(tensor_tree, np.ndarray):
        tensor_tree = th.as_tensor(tensor_tree)
    if isinstance(tensor_tree, (float, int)):
        tensor_tree = th.as_tensor(tensor_tree)
    if isinstance(tensor_tree, np.ndarray):
        return gym_spaces.Box(high=(np.ones_like(tensor_tree)*float("+inf")),
                              low=(np.ones_like(tensor_tree)*float("-inf")),
                              dtype=tensor_tree.dtype)
    if isinstance(tensor_tree, th.Tensor):
        return ThBox(high=(th.ones_like(tensor_tree)*float("+inf")).cpu().numpy(),
                    low=(th.ones_like(tensor_tree)*float("-inf")).cpu().numpy(),
                    dtype=torch_to_numpy_dtype_dict[tensor_tree.dtype],
                    torch_device=tensor_tree.device,
                    labels=labels)
    else:
        raise RuntimeError(f"Unexpected tree element type {type(tensor_tree)}")
    

def sizetree_from_space(space : gym_spaces.Space):
    if isinstance(space, gym_spaces.Dict):
        return {k : sizetree_from_space(v) for k,v in space.spaces.items()}
    elif isinstance(space, gym_spaces.Box):
        return space.shape
    else:
        raise NotImplemented(f"space {space} is not supported.")
    
def is_leaf_finite(tensor : th.Tensor | np.ndarray):
    if isinstance(tensor, th.Tensor):
        return th.all(th.isfinite(tensor))
    elif isinstance(tensor, (np.ndarray, int, float)):
        return np.all(np.isfinite(tensor))
    else:
        return True
        # raise NotImplementedError(f"Unsupported type {type(tensor)}")

def is_all_finite(tree : TensorTree):
    tree = flatten_tensor_tree(tree)
    tree : dict[Any, th.Tensor] = map_tensor_tree(tree, lambda l: th.as_tensor(l))
    is_finites = map_tensor_tree(tree, is_leaf_finite)
    return th.all(th.stack(list(is_finites.values())))

def non_finite_flat_keys(tree : TensorTree):
    return [k for k,v in flatten_tensor_tree(map_tensor_tree(tree, is_leaf_finite)).items() if not v]

def is_leaf_bounded(tensor : th.Tensor | np.ndarray | float,
                    min : th.Tensor | np.ndarray | float,
                    max : th.Tensor | np.ndarray | float):
    if isinstance(tensor, th.Tensor):
        return th.logical_and(th.all(tensor >= min), th.all(tensor <= max))
    elif isinstance(tensor, np.ndarray):
        return np.all(tensor >= min) and np.all(tensor <= max)
    else:
        raise NotImplementedError(f"Unsupported type {type(tensor)}")

def is_all_bounded(tree : TensorTree,
                    min : th.Tensor | np.ndarray | float,
                    max : th.Tensor | np.ndarray | float):
    r = flatten_tensor_tree(map_tensor_tree(tree, lambda t: is_leaf_bounded(t,min=min,max=max))).values()
    # print(f"r = {r}")
    return all(r)


def to_contiguous_tensor(value):
    if isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value)
    return th.as_tensor(value)