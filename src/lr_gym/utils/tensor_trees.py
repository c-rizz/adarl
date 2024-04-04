from lr_gym.utils.spaces import gym_spaces
import torch as th
from typing import Optional, Any
import numpy as np
from lr_gym.utils.utils import torch_to_numpy_dtype_dict


def create_tensor_tree(batch_size : int, space : gym_spaces.Space, share_mem : bool, device : th.device) -> th.Tensor | dict[Any, th.Tensor]:
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

def fill_tensor_tree(env_idx : Optional[int], src_tree : dict | th.Tensor, dst_tree : dict | th.Tensor, depth = 0):
    if isinstance(src_tree, dict):
        if not isinstance(dst_tree,dict):
            raise RuntimeError(f"Tree element type mismatch. src = {type(src_tree)}, dst = {dst_tree}")
        if src_tree.keys() != dst_tree.keys():
            raise RuntimeError(f"source and destination keys do not match: src={src_tree.keys()} dst={dst_tree.keys()}")
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

def map_tensor_tree(src_tree : dict | th.Tensor, func):
    if isinstance(src_tree, dict):
        r = {}
        for k in src_tree.keys():
            r[k] = map_tensor_tree(src_tree[k], func = func)
        return r
    elif isinstance(src_tree, tuple):
        return tuple([map_tensor_tree(e, func = func) for e in src_tree])
    elif isinstance(src_tree, list):
        return [map_tensor_tree(e, func = func) for e in src_tree]
    else:
        return func(src_tree)

def flatten_tensor_tree(src_tree : dict | th.Tensor) -> dict:
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



def unstack_tensor_tree(src_tree : dict | th.Tensor) -> list:
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