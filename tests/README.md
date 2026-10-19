# adarl test suite

Pytest-based tests for adarl. The suite is always runnable: anything needing a GPU or an
optional backend skips itself when the requirement is missing.

## Layout

```
tests/
  conftest.py                     # layout docs; markers declared in pyproject.toml
  utils/                          # pure unit tests (fast, CPU, no simulator)
    test_quaternion.py
    test_masked_tensor_ops.py
    test_normalization.py
    test_robot_helpers.py         # pinocchio Robot: construction, FK, multi-model merge, pickling, collisions
    test_obs_converter_passthrough.py # ObsConverter passthrough components (default path unchanged)
    test_observation_filter.py    # ObservationFilter: vec/img split, named filters, concat spaces+labels
    test_vec_state_helper.py      # ThBoxStateHelper rolling history buffer: eager semantics, torch.compile(fullgraph), cudagraph feedback hazard (GPU)
  adapters/
    adapter_compliance.py         # shared, backend-agnostic behavioral checks (not a test module)
    _backends.py                  # capability detection, skip decorators, adapter builders
    test_mujoco_compliance.py     # MuJoCo (CPU) — full compliance, green
    test_mjx_compliance.py        # MJX jax+warp — xfail (see Findings)
    test_mjx_static_link_warp.py  # regression: moving a fixed link under warp (green)
    test_set_rows_cols_masks.py   # unit: masked sub-block writer (sparse-mask regression)
    test_impulse_persistence.py   # regression: per-env impulse persists & is env-isolated
    test_mjx_rendering.py         # warp RGB+depth: shapes/dtypes, depth content, vec_mask, out= path
    test_mjx_impedance_queue.py   # impedance command queue: bounded under stepping, delay, drain, fill
    test_mjx_per_env_run.py       # per-env run() durations: frozen envs keep state/clock/stats/filters, clock past float32 limit
    test_camzmq_cameras.py        # real-robot camera images via camzmq: sim contract, invalid depth -> 0, camera_info, stale frame
    test_th_jax_interop.py        # torch<->jax DLPack boundary: view vs copy, donation guard, stream ordering, clock read
    test_link_velocity_consistency.py # link velocity vs a finite difference of its own position; step stats bracket the state
  envs/                           # env-level bookkeeping, on a minimal CPU env (no simulator)
    test_episode_bookkeeping.py   # episodes end at their own step limit, counters reset, mean length matches the limits
    test_recorder_roundtrip.py    # the recorder snapshots what it is handed; the saved file is the per-step series
benchmarks/                       # perf/demo scripts (NOT collected by pytest)
  bench_mjx_adapter.py
  bench_genesis_adapter.py
```

`adapter_compliance.py` is the reusable heart of the adapter tests: a single
`run_vec_adapter_compliance_test(adapter)` exercises introspection, state shapes/devices,
direct state setting (masked + unmasked), gravity/ballistics, time bookkeeping, effort,
joint-impedance semantics (immediate/queued/delayed/masked), step statistics, collision
monitoring and world reset — against *any* `BaseVecSimulationAdapter`.

## Running

The project venv is `…/virtualenv/host313`. `pytest` is in `[project.optional-dependencies].test`.

```bash
cd <repo root>                       # dir with pyproject.toml
python -m pytest                     # everything
python -m pytest -m "not gpu"        # fast CPU lane (unit tests + MuJoCo)
python -m pytest -m gpu              # only GPU-backed adapter tests
python -m pytest tests/utils         # just the unit tests
python -m pytest -m mjx              # just the MJX adapter tests
```

Markers: `gpu`, `slow`, `mujoco`, `mjx`, `genesis`, `integration` (`--strict-markers` is on).

## Findings

Wiring the existing checks into a real suite surfaced several latent bugs.

**Fixed as part of this work**

1. `MjxAdapter`: `_forward_needed` was only created inside `run()`, so the first
   `getJointsState`/`getLinksState` after `startup()` (with no prior state-set) raised
   `AttributeError`. `startup()` now marks a forward pending so the first read forwards and
   refreshes the monitored-data cache.
2. `utils/dbg/dbg_checks.py`: `dbg_check`/`dbg_run` read `default_session.debug_level`, which
   only exists after `Session.initialize()`. Any bare use (e.g. `masked_to_masked_assign` in a
   unit test) crashed. Now defaults to off via `getattr(default_session, "debug_level", 0)`.
3. `MjxJointImpedanceAdapter.set_impedance_controlled_joints`: an empty joint list built the
   controlled-joint id array with `np.array([])`/`jnp.array([])`, which default to **float**.
   Since those ids index joint-state arrays, `setJointsStateDirect` (via the filter reset) and
   `step()` (via impedance-command application) crashed with a `float32[0]` indexer whenever no
   impedance joint had been configured yet. Now forced to int dtype — fixes both the
   setJointsStateDirect crash and the "must configure impedance joints before stepping" symptom.
4. `MjxJointImpedanceAdapter.control_period()` was missing its `return` (returned `None`).
5. `MjxJointImpedanceAdapter.get_current_joint_impedance_command()` returned
   `self._last_applied_jimp_cmd`, which was never assigned (always `AttributeError`). Now
   reconstructs the currently-active command from the command queue at the current sim time
   (`_peek_current_impedance_cmd`) — the same selection a step makes, without consuming the queue;
   envs with no active command return a zero command. Handles the immediate/delayed/masked cases.

6. `set_rows_cols_masks` (MjxAdapter): with a **sparse boolean** mask and full-size `vals` it
   resolved the mask via `jnp.nonzero(mask, size=vec_size)`, which pads missing entries with index
   **0** — clobbering row 0 (env 0) and misassigning per-row values. This corrupted env-0 impulses
   every step (impulses vanished after ~1 frame in env-0 videos, regardless of duration) and was a
   latent hazard for the effort/impedance paths when their mask went sparse. Reworked to a proper
   masked overwrite (boolean dims overwrite only `True` rows, keep the rest; integer dims stay a
   compacted cross-product write). Guarded by `test_set_rows_cols_masks.py` and
   `test_impulse_persistence.py`.
7. `MjxAdapter` impulse `impulse_startends_stime` was initialised with `fill_value=-1` → **int32**,
   so impulse start/end **times were truncated to whole seconds** (sub-second impulses collapsed to
   a zero-length window and never fired). Now `float32`.

8. `EnvRunner(env)` could not be constructed with its default arguments: `ui_render_envs` defaults to
   `[]`, and `th.as_tensor([])` is a **float** tensor, which `EnvRunnerInterface` then uses as an index
   (`IndexError: tensors used as indices must be long, int, byte or bool`). Now built with
   `dtype=th.long`.
9. `Async_cuda2cpu_queue`'s worker thread was not a daemon, so it deadlocked interpreter exit: Python
   joins non-daemon threads *before* running `atexit` handlers, so the queue's own `atexit` `close()`
   could never run and any process using it hung at exit. Now `daemon=True`.

10. `MujocoAdapter` reported mjx's com-based `cvel[:,3:6]` as the link linear velocity, without
    shifting it to the point the pose beside it refers to — the same defect `c67d8ea` fixed for
    `MjxAdapter`, found by `test_link_velocity_consistency.py` the first time it ran. Both readers
    (`_read_links_state`, which serves `getLinksState`, and `_read_link_velocities`, which feeds the
    step statistics) now go through `_links_frame_velocities`:
    `com_linvel + omega x (point - subtree_com[body_rootid])`, with `point` being `xipos` or `xpos`
    according to the pose requested. The angular part needs no shift.

**Platform bug, worked around (not an adarl defect)**

- **torch 2.10 miscompiles a cudagraph-compiled function whose inputs alias one tensor.**
  `RobotVecEnv._get_adapter_data_raw_mjx` passed slices of one link 13-vector
  (`[:,0:2]`, `[:,2]`, `[:,3:7]`, `[:,7:10]`, `[:,10:13]`) to `_get_new_instantaneous_state`, which is
  `th_compile_ext(copy_outs=True, mode="max-autotune", fullgraph=True)`. In real 8192-env runs the
  compiled function returned the **body position** for the absolute linear velocity, the absolute
  angular velocity and the height alike, from inputs verified correct immediately before: the aliased
  views were read wrongly. Velocities read as metres of position, every velocity-derived reward and
  the recorded episodes were nonsense, while the adapter, the state helper and the recorder were all
  correct. Fixed either by cloning the slices (done, in both adapter-data paths) or by torch >= 2.14.
  `tests/envs/test_compiled_state_builder.py` pins the contract; it passes on 2.14, so it guards the
  platform rather than the env code.
  Worth knowing for the next hunt of this shape: the corruption only ever appeared in full training
  runs. Single-env replays, two-env replays, 64-env probes and the adapter tests were all clean on the
  very machine that failed, which is what made it look machine-specific for a day.

With 1–10 fixed, both MuJoCo and MJX (jax + warp) pass full compliance; there are no open findings in
adarl itself.

**Documented hazard (characterised, not a code bug)**

- `ThBoxStateHelper.update()`'s rolling history buffer (`state.roll(1, dims=1)`, newest at index
  0) is correct in eager and captures as a single graph under `torch.compile(fullgraph=True)`.
  But under `mode="max-autotune"` on **CUDA** (which turns on cudagraphs) the per-step loop
  `state = update(sample, state)` feeds a *static cudagraph output buffer* straight back in as the
  next input, so a later run overwrites a buffer the graph still owns. This has two faces, same
  root cause: current torch's cudagraph-tree checker *raises*
  (`RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`);
  on versions/configs without that checker it corrupts *silently* -- stale reads → wrong values →
  the non-finite states this originally showed up as.
  This is the concrete reason the `@th.compile(mode="max-autotune", ..., fullgraph=True)` on
  `RobotVecEnv._post_step_optimized` is commented out. Fix if re-enabling: **clone the compiled
  output** before feeding it back (`cudagraph_mark_step_begin()` alone is insufficient -- the
  fed-back input is the overwritten one). `test_vec_state_helper.py` pins both the failure and the
  clone remedy (GPU-only tests; skipped without CUDA).

**By design (not a bug)**

- `getJointsState`/`getLinksState` are only required to work for *monitored* elements. Some
  adapters (MuJoCo) also serve non-monitored ones, but that is not part of the contract, so the
  compliance checks monitor everything they read back.

## Roadmap — tests worth adding

Unit (fast, high value, no hardware):
- `tensor_trees` (map/flatten/stack/cat), `tensor_struct`, `running_mean_std`, `spaces`,
  `ObsConverter`, replay buffers (`ThVecDictEpReplayBuffer`).
- More geometry: `quat_swing_twist_decomposition_xyzw`, `average_two_quaternions`,
  `ros_rpy_to_quaternion_xyzw`, `quaternion_xyzw_from_rotmat` round-trips.
- `sample_distr` / distribution-def sampling (shapes, determinism with a seeded generator).

Adapters:
- Genesis compliance test (same pattern, `@requires_genesis`).
- Parametrize compliance over `vec_size` (1 and N) once the MJX gaps are fixed.
- Determinism: same seed ⇒ identical rollout.
- Domain randomization (`alter_model`): mass/friction/damping changes actually change dynamics,
  per-env, jax **and** warp.
- Rendering smoke: rgb/depth shapes for both `render_backend="cpu"` and `"warp"`.
- `setJointsAndLinksStateDirect` combined path; partial `vec_mask` reset correctness.

Envs / integration:
- `GraspVecEnv` reset+step smoke (obs/action space shapes, finite rewards) at small vec size.
- `lr_wrappers` observation/action transforms.
- A short training smoke (a few PPO iters) to catch cross-component breakage.

Infra:
- A CI config running the `-m "not gpu"` lane on every push and the `gpu`/`slow` lane nightly.
