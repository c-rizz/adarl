import torch.multiprocessing as mp
from typing import Literal

_context = None
_context_type = ""
_manager = None

def get_context(method : Literal["fork","spawn","forkserver"] = "forkserver"):
    global _context
    global _context_type
    if _context is None:
        _context = mp.get_context(method=method)
        _context_type = method
    if _context_type != method:
        raise RuntimeError(f"Can only use one mp method at a time")
    return _context

def get_manager(method : Literal["fork","spawn","forkserver"] = "forkserver"):
    ctx = get_context(method)
    global _manager
    if _manager is None:
        _manager = ctx.Manager()
    return _manager