import torch.multiprocessing as mp
from typing import Literal


def get_context(method : Literal["fork","spwan","forkserver"] = "forkserver"):
    return mp.get_context(method=method)