from __future__ import annotations
from lr_gym.utils.spaces import gym_spaces
import torch as th
from typing import Optional, Any, List, Dict, Tuple, Union, Callable, TypeVar
import numpy as np
from lr_gym.utils.utils import torch_to_numpy_dtype_dict
import dataclasses

LeafType = TypeVar("LeafType")
TensorTree = Union[Dict[Any,"TensorTree[LeafType]"],
                   List["TensorTree[LeafType]"],
                   Tuple["TensorTree[LeafType]", ...],
                   LeafType]

def create_tensor_tree(batch_size : int, space : gym_spaces.Space, share_mem : bool, device : th.device) -> Union[th.Tensor, Dict[Any, th.Tensor]]:
    if isinstance(space, gym_spaces.Dict):
        tree = {}
        for k, s in space.spaces.items():
            tree[k] = create_tensor_tree(batch_size, s, share_mem, device)
        return tree
    elif isinstance(space, gym_spaces.Box):
        thdtype = th.as_tensor(space.sample()).dtype
        t = th.zeros(size=(batch_size,)+space.shape, dtype=thdtype, device = device)
        if share_mem:
            t.share_memory_()
        return t
    else:
        raise RuntimeError(f"Unsupported space {space}")

def fill_tensor_tree(env_idx : Optional[int], src_tree : TensorTree, dst_tree : TensorTree, depth = 0):
    if isinstance(src_tree, dict):
        if not isinstance(dst_tree,dict):
            raise RuntimeError(f"Tree element type mismatch. src = {type(src_tree)}, dst = {dst_tree}")
        if src_tree.keys() != dst_tree.keys():
            src_keys = set(src_tree.keys())
            dst_keys = set(dst_tree.keys())
            raise RuntimeError(f"source and destination keys do not match:\nsrc={src_keys}\ndst={dst_keys}\ndiff={(src_keys-dst_keys).union(dst_keys-src_keys)}")
        for k in dst_tree.keys():
            fill_tensor_tree(env_idx, src_tree[k], dst_tree[k], depth = depth+1)
        if depth == 0:
            th.cuda.synchronize() # sync non-blocking copies
    elif isinstance(src_tree, th.Tensor):
        if not isinstance(dst_tree,th.Tensor):
            raise RuntimeError(f"Tree element type mismatch. src = {type(src_tree)}, dst = {dst_tree}")
        if env_idx is not None:
            dst_tree[env_idx].copy_(src_tree, non_blocking=True)
        else:
            dst_tree.copy_(src_tree, non_blocking=True)
    else:
        raise RuntimeError(f"Unexpected tree element type {type(src_tree)}")

T = TypeVar('T')
U = TypeVar('U')
def map_tensor_tree(src_tree : TensorTree[U], func : Callable[[TensorTree[U]],TensorTree[T]]) -> TensorTree[T]:
    if isinstance(src_tree, dict):
        r = {}
        for k in src_tree.keys():
            r[k] = map_tensor_tree(src_tree[k], func = func)
        return r
    elif isinstance(src_tree, tuple):
        return tuple([map_tensor_tree(e, func = func) for e in src_tree])
    elif isinstance(src_tree, list):
        return [map_tensor_tree(e, func = func) for e in src_tree]
    elif dataclasses.is_dataclass(src_tree):
        mapped_fields = {field.name: map_tensor_tree(getattr(src_tree, field.name), func = func)
                          for field in dataclasses.fields(src_tree)}
        return src_tree.__class__(**mapped_fields)
    else:
        return func(src_tree)

def flatten_tensor_tree(src_tree : TensorTree) -> dict:
    if isinstance(src_tree, tuple):
        src_tree = {f"T{i}":src_tree[i] for i in range(len(src_tree))}
    elif isinstance(src_tree, list):
        src_tree = {f"L{i}":src_tree[i] for i in range(len(src_tree))}
    if isinstance(src_tree, dict):
        r = {}
        for k in src_tree.keys():
            subdict = flatten_tensor_tree(src_tree[k])
            for sk,sv in subdict.items():
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




def unstack_tensor_tree(src_tree : TensorTree) -> list:
    if isinstance(src_tree, dict):
        # print(f"src_tree = {src_tree}")
        dictlist = []
        for k in src_tree.keys():
            unstacked_subtree = unstack_tensor_tree(src_tree[k])
            stack_size = len(unstacked_subtree)
            if len(dictlist)==0:
                dictlist = [{} for _ in range(stack_size)]
            for i in range(stack_size):
                dictlist[i][k] = unstacked_subtree[i]
        # print(f"src_tree = {src_tree}, unstacked = {dictlist}")
        return dictlist
    elif isinstance(src_tree, th.Tensor):
        ret = src_tree.unbind()
        # print(f"src_tree = {src_tree}, unstacked = {ret}")
        return ret
    else:
        raise RuntimeError(f"Unexpected tree element type {type(src_tree)}")

def space_from_tree(tensor_tree):
    if isinstance(tensor_tree, dict):
        subspaces = {}
        for k in tensor_tree.keys():
            subspaces[k] = space_from_tree(tensor_tree[k])
        return gym_spaces.Dict(subspaces)
    if isinstance(tensor_tree, np.ndarray):
        tensor_tree = th.as_tensor(tensor_tree)
    if isinstance(tensor_tree, (float, int)):
        tensor_tree = th.as_tensor(tensor_tree)
    if isinstance(tensor_tree, th.Tensor):
        return gym_spaces.Box(high=(th.ones_like(tensor_tree)*float("+inf")).cpu().numpy(),
                              low=(th.ones_like(tensor_tree)*float("-inf")).cpu().numpy(),
                              dtype=torch_to_numpy_dtype_dict[tensor_tree.dtype])
    else:
        raise RuntimeError(f"Unexpected tree element type {tensor_tree}")