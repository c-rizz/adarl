import torch.multiprocessing as mp
from typing import Literal

context = None
context_type = ""
def get_context(method : Literal["fork","spwan","forkserver"] = "forkserver"):
    global context
    global context_type
    if context is None:
        context = mp.get_context(method=method)
        context_type = method
    if context_type != method:
        raise RuntimeError(f"Can only use one mp method at a time")
    return context