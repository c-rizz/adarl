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
  adapters/
    adapter_compliance.py         # shared, backend-agnostic behavioral checks (not a test module)
    _backends.py                  # capability detection, skip decorators, adapter builders
    test_mujoco_compliance.py     # MuJoCo (CPU) — full compliance, green
    test_mjx_compliance.py        # MJX jax+warp — xfail (see Findings)
    test_mjx_static_link_warp.py  # regression: moving a fixed link under warp (green)
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

With 1–5 fixed, both MuJoCo and MJX (jax + warp) pass full compliance; there are no open findings.

**By design (not a bug)**

- `getJointsState`/`getLinksState` are only required to work for *monitored* elements. Some
  adapters (MuJoCo) also serve non-monitored ones, but that is not part of the contract, so the
  compliance checks monitor everything they read back.

## Roadmap — tests worth adding

Unit (fast, high value, no hardware):
- `tensor_trees` (map/flatten/stack/cat), `tensor_struct`, `running_mean_std`, `spaces`,
  `ObsConverter`, `vec_state_helper`, replay buffers (`ThVecDictEpReplayBuffer`).
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
