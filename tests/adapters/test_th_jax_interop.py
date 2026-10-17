"""The torch <-> jax boundary: aliasing and ordering.

adarl passes buffers between torch and jax through DLPack, with no copy, and the jitted adapter calls
donate the simulation state so jax can write the next state over the previous one. That combination has
two hazards, and both have bitten:

  - a torch tensor obtained from `jax2th` is a *view* of a jax buffer. A later donating call may
    overwrite that buffer, so the view silently changes value, or shows a buffer that now belongs to a
    different array. `jax2th_copy` exists for anything that outlives the call.
  - a copy or a read is not enough on its own: torch and jax use different CUDA streams, so a torch
    kernel reading a DLPack view can be scheduled after jax has already overwritten it, and torch
    memory handed to jax can be freed by torch (and handed to the next torch allocation) while the jax
    kernels are still reading it. `jit_th_synced` and `jax2th_copy` order the two sides without a host
    synchronisation.

These tests exercise those guarantees directly, under allocation pressure, since the failures only show
when an allocator actually recycles the block in question. They are written so that a *correct*
implementation passes: a failure means the ordering guarantee was lost, not that the test is flaky.

The last test is the adapter-level regression: the per-env substep counter is the smallest buffer in
the simulation state and is read through `jax2th_copy` on every step, which is exactly the pattern that
produced a corrupted clock (counts off by whole powers of two) in loco_kyon_ppo.
"""
import pytest
import torch as th

from _backends import requires_jax_gpu, requires_mjx_warp, _log_folder

pytestmark = [pytest.mark.gpu]

ITERS = 200          # enough iterations for an allocator to recycle a freed block
BIG = 1 << 18        # floats per "state" array: big enough to go through the caching allocator
SMALL = 8            # the counter: vec_size-sized, the shape that got corrupted in the wild


def _churn_torch(device, n: int = 4):
    """Allocate and free torch blocks, so a block freed by the other side gets handed out again."""
    for _ in range(n):
        t = th.empty((BIG,), dtype=th.float32, device=device)
        t.fill_(-1.0)
        del t


@pytest.fixture(scope="module")
def jx():
    jax = pytest.importorskip("jax")
    return jax


@requires_jax_gpu
def test_jax_refuses_to_donate_a_buffer_torch_still_views(jx):
    """A live DLPack view blocks donation, which is a guarantee worth knowing about.

    So the hazard is not "a donating call silently overwrites a view the caller still holds": jax
    rejects that outright. It is the weaker one in the next test, where nothing holds a reference any
    more and the allocator is free to recycle the block.
    """
    import jax.numpy as jnp
    from adarl.adapters.MjxAdapter import jax2th

    dev = th.device("cuda", 0)
    arr = jnp.arange(SMALL, dtype=jnp.int32)
    view = jax2th(arr, dev)                     # noqa: F841  (kept alive on purpose)

    def _bump(a):
        return a + 1000
    bump = jx.jit(_bump, donate_argnums=(0,))

    with pytest.raises(Exception) as excinfo:
        jx.block_until_ready(bump(arr))
    assert "donation" in str(excinfo.value).lower() or "external reference" in str(excinfo.value).lower(), \
        f"expected a donation-vs-external-reference error, got: {excinfo.value}"


@requires_jax_gpu
def test_jax2th_copy_owns_its_data_when_the_source_is_recycled(jx):
    """A copy must keep its values once jax has dropped, and reused, the buffer it came from."""
    import jax.numpy as jnp
    from adarl.adapters.MjxAdapter import jax2th_copy

    dev = th.device("cuda", 0)
    arr = jnp.arange(SMALL, dtype=jnp.int32)
    copy = jax2th_copy(arr, dev)
    del arr                                     # jax may now free and reuse that block
    recycled = [jnp.full((SMALL,), 7777, dtype=jnp.int32) for _ in range(64)]
    jx.block_until_ready(recycled)
    th.cuda.synchronize()
    assert th.equal(copy.cpu(), th.arange(SMALL, dtype=th.int32)), \
        "jax2th_copy must own its data and survive the source buffer being recycled"


@requires_jax_gpu
def test_jax2th_copy_is_not_outrun_by_jax_freeing_the_buffer(jx):
    """A copy is only *enqueued*, so the source buffer has to stay alive until it has happened.

    This is the failure that corrupted the clock in loco_kyon_ppo. jax2th hands torch a view, the copy
    torch makes of it is queued on torch's stream, and dropping the view frees the buffer on the host
    right away. jax's allocator is ordered against jax's stream alone, so it hands the block to the next
    jax op -- below a boolean array, whose 0x01 bytes read back as 0x01010100 in the int32 substep
    counter: a jump of 2**24 counts, 32768 s of simulated time at 1/512 s per substep.
    Torch's stream is loaded with work first, so the queued copy really is still pending when the
    recycling happens; without that the read usually wins the race and the bug hides.
    """
    import jax.numpy as jnp
    from adarl.adapters.MjxAdapter import jax2th_copy

    dev = th.device("cuda", 0)
    truth = 300
    a = th.randn((2048, 2048), device=dev)
    b = th.randn((2048, 2048), device=dev)
    for trial in range(3):
        counter = jnp.full((SMALL,), truth, dtype=jnp.int32)
        for _ in range(60):
            a = a @ b                               # keep torch's stream busy ahead of the read
        got = jax2th_copy(counter, dev).to(th.float32)
        del counter                                 # nothing references the jax buffer any more
        recycled = [jnp.full((SMALL, 4), True) for _ in range(300)]  # bools land on the freed block
        jx.block_until_ready(recycled)
        values = got.cpu().to(th.int64).tolist()
        assert all(v == truth for v in values), \
            f"trial {trial}: counter read back as {[hex(v) for v in values]} instead of {truth}: jax " \
            f"recycled the buffer before torch's queued copy of it had run"


@requires_jax_gpu
def test_jax2th_copy_survives_donation_under_allocation_pressure(jx):
    """Read a small counter out of a donated state every iteration; every read must stay valid.

    This is the unit-level shape of the adapter's per-env substep counter: a tiny int32 array living in
    a donated state next to much bigger buffers, read on every step and kept by the caller.
    """
    import jax.numpy as jnp
    from adarl.adapters.MjxAdapter import jax2th_copy

    dev = th.device("cuda", 0)

    def _advance(counter, payload):
        return counter + 1, payload * 1.0000001 + 1.0
    advance = jx.jit(_advance, donate_argnums=(0, 1))

    counter = jnp.zeros((SMALL,), dtype=jnp.int32)
    payload = jnp.ones((BIG,), dtype=jnp.float32)
    kept = []
    for i in range(ITERS):
        counter, payload = advance(counter, payload)
        kept.append(jax2th_copy(counter, dev))
        _churn_torch(dev)

    th.cuda.synchronize()
    for i, t in enumerate(kept):
        expected = i + 1
        got = t.cpu()
        assert th.equal(got, th.full((SMALL,), expected, dtype=th.int32)), \
            f"read {i} came back as {got.tolist()} instead of {expected}: the copy was not ordered " \
            f"before jax reused the donated buffer"


@requires_jax_gpu
def test_counter_reads_survive_cudagraph_torch_traffic(jx):
    """The same counter reads, but with torch running CUDA graphs alongside.

    The envs run torch.compile'd code in cudagraph mode around the adapter, which allocates from a
    separate graph pool and replays kernels on its own stream. That is the one ingredient the
    adapter-level tests do not have, and the env-level reproduction of the corrupted clock does.
    """
    import jax.numpy as jnp
    from adarl.adapters.MjxAdapter import jax2th_copy

    dev = th.device("cuda", 0)

    def _advance(counter, payload):
        return counter + 1, payload * 1.0000001 + 1.0
    advance = jx.jit(_advance, donate_argnums=(0, 1))

    @th.compile(mode="reduce-overhead")
    def torch_work(x):
        return (x * 2.0 + 1.0).sin().sum()

    counter = jnp.zeros((SMALL,), dtype=jnp.int32)
    payload = jnp.ones((BIG,), dtype=jnp.float32)
    buf = th.ones((BIG,), dtype=th.float32, device=dev)
    for i in range(ITERS):
        counter, payload = advance(counter, payload)
        got = jax2th_copy(counter, dev).cpu()
        torch_work(buf)
        _churn_torch(dev, n=2)
        assert th.equal(got, th.full((SMALL,), i + 1, dtype=th.int32)), \
            f"read {i} came back as {got.tolist()} instead of {i+1}"


@requires_jax_gpu
def test_th2jax_input_is_not_recycled_while_jax_reads_it(jx):
    """The other direction: torch memory handed to jax must not be reused while the call still reads it.

    Torch's caching allocator only tracks torch's own streams, so without ordering it hands the block to
    the next torch allocation, which overwrites what the jax kernels are about to read.
    """
    import jax.numpy as jnp
    from adarl.adapters.MjxAdapter import th2jax, jax2th_copy

    dev = th.device("cuda", 0)
    jax_dev = jx.devices("gpu")[0]

    @jx.jit
    def reduce_sum(a):
        # a few ops, so the call takes long enough for a racing overwrite to land mid-flight
        for _ in range(8):
            a = a * 1.0 + 0.0
        return jnp.sum(a, dtype=jnp.float32)

    for i in range(ITERS // 4):
        src = th.full((BIG,), 2.0, dtype=th.float32, device=dev)
        out = reduce_sum(th2jax(src, jax_dev))
        del src              # torch may hand this block to the next allocation right away
        _churn_torch(dev)    # ... which is what these do
        got = float(jax2th_copy(jnp.asarray(out).reshape(1), dev).cpu()[0])
        assert got == pytest.approx(2.0 * BIG, rel=1e-6), \
            f"iteration {i}: jax summed {got} instead of {2.0*BIG}, its input was overwritten mid-call"


@requires_jax_gpu
def test_jit_th_synced_orders_torch_after_the_call(jx):
    """A jitted call's result must be complete before torch reads it, without a host synchronisation."""
    import jax.numpy as jnp
    from adarl.adapters.MjxAdapter import jit_th_synced, jax2th

    dev = th.device("cuda", 0)

    @jit_th_synced
    def slow_fill(seed):
        a = jnp.full((BIG,), seed, dtype=jnp.float32)
        for _ in range(16):
            a = a * 1.0 + 0.0
        return a

    for i in range(ITERS // 4):
        out = jax2th(slow_fill(jnp.float32(i)), dev)
        # no th.cuda.synchronize() here on purpose: jit_th_synced must have ordered torch's stream
        assert float(out.min()) == pytest.approx(i), f"iteration {i}: torch read the result too early"
        assert float(out.max()) == pytest.approx(i)
        _churn_torch(dev)


@pytest.mark.mjx
@pytest.mark.slow
@requires_jax_gpu
@requires_mjx_warp
@pytest.mark.parametrize("mjx_impl", ["warp"])
def test_adapter_clock_advances_exactly(mjx_impl):
    """Regression: the per-env clock must advance by exactly one step's worth of substeps per step.

    getEnvTimeFromStartup() is substeps*dt read out of the donated simulation state. In loco_kyon_ppo
    this came back corrupted (counts off by 2**24 substeps on a subset of the envs, at a different step
    on every run), which turned into an enormous LAST_STEP_DT and from there into out-of-bounds
    observations and non-finite states. Interleaving torch allocations and state writes, the way a real
    env does on reset, is what makes the recycling happen.
    """
    import jax
    from adapter_compliance import _spawn_defs, CART_JOINT, FOOT_JOINT, BAR_LINK, BALL_LINK
    from adarl.adapters.MjxJointImpedanceAdapter import MjxJointImpedanceAdapter

    VEC, DT, SUBSTEPS = 8, 1 / 512, 10
    dev = th.device("cuda", 0)
    # the impedance adapter, with its reference filter and command queue, is what the envs drive: it
    # adds donating jitted calls around every step, which is the traffic the clock read has to survive
    adapter = MjxJointImpedanceAdapter(vec_size=VEC, enable_rendering=False, jax_device=jax.devices("gpu")[0],
                                       output_th_device=dev, sim_step_dt=DT, step_length_sec=SUBSTEPS * DT,
                                       realtime_factor=-1, show_gui=False, mjx_impl=mjx_impl,
                                       render_backend="cpu", log_folder=_log_folder(),
                                       max_joint_impedance_ctrl_torques={CART_JOINT: 100.0, FOOT_JOINT: 100.0},
                                       reference_filter_mode="second_order")
    adapter.build_scenario(_spawn_defs())
    adapter.set_monitored_joints([CART_JOINT, FOOT_JOINT])
    adapter.set_monitored_links([BAR_LINK, BALL_LINK])
    adapter.set_impedance_controlled_joints([CART_JOINT])
    adapter.startup()
    try:
        prev = adapter.getEnvTimeFromStartup().clone()
        for i in range(ITERS):
            cmd = th.zeros((VEC, 1, 5), dtype=th.float32, device=dev)
            cmd[:, 0, 0] = 0.2 * ((i % 10) - 5)   # a moving position reference
            cmd[:, 0, 3] = 200.0                  # kp
            cmd[:, 0, 4] = 10.0                   # kd
            adapter.setJointsImpedanceCommand(cmd, delay_sec=0.0)
            adapter.step()
            # the env does this kind of work between steps: reads of the state and torch allocations
            adapter.getLinksState(requestedLinks=[BAR_LINK, BALL_LINK])
            adapter.getJointsState(requestedJoints=[CART_JOINT, FOOT_JOINT])
            _churn_torch(dev, n=2)
            # ... and, when episodes end, donating writes of the state on a changing subset of envs,
            # which is the pattern that preceded the corrupted clock in the wild
            if i % 7 == 0:
                mask = th.zeros((VEC,), dtype=th.bool, device=dev)
                mask[(i // 7) % VEC] = True
                jstate = th.zeros((VEC, 2, 3), dtype=th.float32, device=dev)
                jstate[:, 0, 0] = 0.1 * (i % 5)
                adapter.setJointsStateDirect([CART_JOINT, FOOT_JOINT], jstate, vec_mask=mask)
                _churn_torch(dev, n=2)
            now = adapter.getEnvTimeFromStartup()
            delta = (now - prev).cpu()
            expected = SUBSTEPS * DT
            assert th.allclose(delta, th.full((VEC,), expected, dtype=delta.dtype), atol=1e-6), \
                f"step {i}: clock advanced by {delta.tolist()} instead of {expected} on every env"
            prev = now.clone()
    finally:
        adapter.destroy_scenario()
