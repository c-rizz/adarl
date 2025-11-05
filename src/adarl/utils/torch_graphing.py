import torch as th
from time import monotonic
from typing import Callable

graphed_funcs : dict[tuple[Callable, tuple[tuple[int,...], ...]], Callable] = {}
def graphit(disable=False):
    if not disable:
        def deferred_graphing_decorator(func):
            # This takes thte function and replaces it with a function that on the first call graphs it with the input sizes
            def graph_and_run(func, *args):
                flatten_args = th.utils._pytree.arg_tree_leaves(*args)
                args_sizes = tuple(v.size() for v in flatten_args)
                func_and_sizes = (func, args_sizes)
                graphed_func = graphed_funcs.get(func_and_sizes, None)
                if graphed_func is None:
                    print(f"Graphing function {func} with args sizes {list(args_sizes)}")
                    graphed_func : Callable = th.cuda.make_graphed_callables(func, args)
                    graphed_funcs[func_and_sizes] = graphed_func
                return graphed_func(*args)
            return lambda *args: graph_and_run(func, *args)
        return deferred_graphing_decorator
    else:
        def nope_decorator(func):
            return func
        return nope_decorator

def test():
    use_graph = True
    iterations = 100_000
    enc = th.nn.Sequential(
        th.nn.Linear(6, 128),
        th.nn.ReLU(),
        th.nn.Linear(128, 2)
    ).to("cuda")

    dec = th.nn.Sequential(
        th.nn.Linear(2, 128),
        th.nn.ReLU(),
        th.nn.Linear(128, 6)
    ).to("cuda")

    @graphit(disable=not use_graph)
    def autoencode(x, enc_state, dec_state):
        z = th.func.functional_call(enc, enc_state, (x,))
        x_recon = th.func.functional_call(dec, dec_state, (z,))
        return x_recon, z

    optimizer = th.optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=1e-3)
    enc_params = dict(enc.named_parameters())
    dec_params = dict(dec.named_parameters())

    # if use_graph:
    #     example_batch_input = th.randn(128, 4, 6).to("cuda")
    #     maybe_graphed_ae = th.cuda.make_graphed_callables(autoencode, (example_batch_input, enc_params, dec_params))
    # else:
    #     maybe_graphed_ae = autoencode

    def train_step(x):
        x_recon, z = autoencode(x, enc_params, dec_params)
        loss = th.nn.functional.mse_loss(x_recon, x)
        return loss

    dataset = th.randn(1024*1024, 4, 6, device="cuda")
    losses = th.empty(size=(iterations,), device="cuda")
    t0 = monotonic()
    t_prev = t0
    for i in range(iterations):
        optimizer.zero_grad()
        idx = th.randint(0, dataset.size(0), (128,), device="cuda")
        loss = train_step(dataset[idx])
        loss.backward()
        optimizer.step()
        losses[i] = loss
        if i % 1000 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}, Time per iter: {(monotonic()-t_prev)/1000:.6f} s")
            t_prev = monotonic()
    tf = monotonic()
    print(f"Average time per iter: {(tf-t0)/iterations:.6f} s")

    x = th.randn(4, 6).to("cuda")
    y, z = autoencode(x, enc_params, dec_params)
    print(f"Inference = {x} -> {z} -> {y}")


if __name__ == "__main__":
    test()