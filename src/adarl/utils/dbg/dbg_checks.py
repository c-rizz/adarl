from __future__ import annotations
from typing import Callable, Any, Sequence
import torch as th
import adarl.utils.dbg.ggLog as ggLog
import inspect

def get_caller_info():
    frame = inspect.currentframe().f_back  # Get the previous frame (caller)
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    return filename, lineno

printed_dbg_check_msg = False
def dbg_check(is_check_passed : Callable[[],bool|th.Tensor], build_msg : Callable[[],str] | None = None, just_warn : bool = False):
    from adarl.utils.session import default_session
    if default_session.debug_level>0:
        global printed_dbg_check_msg
        if not printed_dbg_check_msg:
            ggLog.warn(f"dbg_check is enabled")
            printed_dbg_check_msg = True
        if not is_check_passed():
            msg = build_msg() if build_msg is not None else f"dbg_check failed"
            if just_warn:
                ggLog.warn(msg)
            else:
                raise RuntimeError(msg)
    # else:
    #     print(f"Dbg check skipped")
    
def dbg_run(func : Callable[[],Any]):
    from adarl.utils.session import default_session
    if default_session.debug_level>0:
        func()

def dbg_check_finite(tensor_tree, min = float("-inf"), max = float("+inf"), async_assert = False):
    from adarl.utils.tensor_trees import is_all_finite, is_all_bounded, flatten_tensor_tree, map_tensor_tree, is_leaf_finite
    if async_assert:
        th._assert_async(is_all_finite(tensor_tree), f"non-finite values in tensor at {get_caller_info()}")
        return
    dbg_check(is_check_passed=lambda: is_all_finite(tensor_tree), 
              build_msg=lambda: f"Non-finite values in tensor tree: isfinite = {map_tensor_tree(flatten_tensor_tree(tensor_tree), is_leaf_finite)}")
    if min != float("-inf") or max != float("+inf"):
        dbg_check(is_check_passed=lambda: is_all_bounded(tensor_tree, min=th.as_tensor(min),max=th.as_tensor(max)), 
              build_msg=lambda: f"out of bounds values in tensor tree: {tensor_tree}")
    
def dbg_check_size(tensor : th.Tensor, size : Sequence[int], msg : str = ""):
    dbg_check(is_check_passed=lambda: tensor.size()==size, 
              build_msg=lambda: f"Unexpected tensor size at {get_caller_info()}: {tensor.size()} instead of {size}. "+msg)