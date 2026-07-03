# Design: System Identification of Centauro with the MuJoCo sysid toolbox

Status: design only (no code yet)
Scope (decided): **fixed base**, identify **joint friction & damping** (frictionloss,
damping, armature/rotor inertia), real-robot I/O through **pyxbot**.
Data collection (decided): **C++-side trajectory-replay-and-record** over a hard-RT loop
(see §5), not a Python control loop.

---

## 1. Building blocks already in the workspace

| Need | What we use | Where |
|------|-------------|-------|
| The sysid engine | `mujoco.sysid` (DeepMind toolbox, ships with MuJoCo 3.6) | site-packages `mujoco/sysid` |
| LM optimizer | `mujoco.sysid.optimize` → `mujoco.minimize.least_squares` / scipy | same |
| URDF for centauro | `compile_xacro_string` on `centauro.urdf.xacro` | [pycentauro/.../xacro_compile_example.py](../../pycentauro/src/pycentauro/xacro_compile_example.py) |
| URDF → MjSpec → MjModel | `aggregate_models`, `add_compiler_options`, `apply_dof_overrides_to_spec` | [mujoco_utils.py](../src/adarl/adapters/mujoco_utils.py) |
| Real-robot logging/commanding | `XbotZmqClient` (`sense()`/`JointState`, `send_command`/`JointsCommand`) | [pyxbot/zmq_client.py](../../xbot2_zmq/pyxbot/src/pyxbot/zmq_client.py) |
| Generic trajectory execution + state recording | **new** pyxbot command + C++ RT handler (§5), sysid-agnostic | xbot2_zmq (to add) |
| Impedance law reference | `τ = kp·(pos_ref−q) + kd·(vel_ref−q̇) + eff_ref` | [MujocoJointImpedanceAdapter.py:255](../src/adarl/adapters/MujocoJointImpedanceAdapter.py#L255) |

**Prereq:** `mujoco.sysid` imports `colorama`, which is currently missing. One-time:
`pip install colorama` in the `host313` venv.

---

## 2. The core idea

The sysid toolbox optimizes model parameters so that a **simulated rollout** matches a
**measured trajectory**. It needs three time-aligned things:

1. **control** — the input driving the system each step (a `sysid.TimeSeries`),
2. **sensordata** — the measured outputs to fit (a `sysid.TimeSeries`),
3. **initial_state** — `mjSTATE_FULLPHYSICS` vector at t₀ (`sysid.create_initial_state`).

These are bundled per experiment into a `sysid.ModelSequences` (a spec + one or more
measured sequences). `sysid.residual` rolls the model out (`sysid_rollout`) and diffs
predicted vs. measured sensordata; `sysid.optimize` runs Levenberg–Marquardt over a
`ParameterDict`.

The whole engineering problem is **making the real robot and the MuJoCo model speak the
same language** — same control representation, same sensor channels, same joint ordering.
With the RT replay-and-record collection of §5, the *time alignment* part is handled at the
source: input and output share the controller's RT clock, so there is no jitter to model
away.

---

## 3. Control representation — the key decision

xbot does **not** apply MuJoCo `position` actuators; the impedance torque is computed in
software (firmware on the real robot; Python at line 255 in the sim adapter). That gives
two clean ways to drive the sysid model, and they identify different things:

### Option A — torque-driven (recommended for friction & damping)
Treat the **measured joint effort** `eff` as the input to a plain per-joint **torque
actuator** (`<motor>` / `general` with `gear=1`). The controller is then completely
removed from the identification loop; we fit only the **passive plant**
(friction, damping, armature, inertia).

- control TimeSeries  = measured `eff` per joint
- sensordata TimeSeries = measured `pos`, `vel` per joint
- Identifies exactly the chosen parameter set, no coupling to K/D estimation.
- Requires that `eff` is a trustworthy measured/commanded torque (centauro reports it in
  `JointState.eff`). If `eff` is motor-side, account for gear ratio.

### Option B — reference-driven (extension, needed only if K/D effective gains matter)
Feed `pos_ref, vel_ref, K, D, τ_ff` and replicate the impedance law inside a
`sysid.CustomRolloutFn` (or via the toolbox's `apply_pdgain` modifier on a position
actuator). This additionally identifies the **effective** K/D the firmware applies.
Out of scope for the first pass, but the recorded data (§5) is a superset (it logs refs +
K/D + eff + pos/vel), so we can switch to B later with the *same* recordings.

**Decision: implement Option A first.** Log everything so B stays available.

---

## 4. Model preparation

1. Compile URDF: `compile_xacro_string(centauro.urdf.xacro, {realsense:false,
   velodyne:false, floating_joint:"false"})` — **floating_joint=false** to get the fixed
   base. Mirror `extra_pkg_paths={"centauro_urdf": ...}` from the example.
2. URDF → MjModel via `aggregate_models([ModelSpawnDef(...)])` from
   [mujoco_utils.py](../src/adarl/adapters/mujoco_utils.py); reuse `add_compiler_options`
   (it sets `inertiafromgeom=false`, radian, strippath) so inertias come from the URDF.
3. Anchor the trunk: with floating_joint=false the root link is already welded to world.
   Physically the robot must be clamped/legs locked to match.
4. Add, **per identified joint**:
   - a torque actuator (`<motor joint=... gear=1/>`),
   - sensors `<jointpos>` and `<jointvel>` (and optionally `<jointactuatorfrc>` for
     cross-checks).
   Add these on the **MjSpec** before compile (consistent with how `aggregate_models`
   already manipulates specs). `TimeSeries.from_control_names` / `from_names` then map our
   column layout onto actuator/sensor indices **by name** — no positional fragility.
5. Set `opt.timestep` equal to the RT control period the data was recorded at (§5), so the
   rollout steps line up 1:1 with the recorded samples.

---

## 5. Data collection — C++ trajectory-replay-and-record (hard-RT)

Rationale: the RT loop runs at a fixed period (typ. 1 kHz) and is jitter-free by
construction. Executing a pre-loaded trajectory and logging in that loop gives input *and*
output on the same RT clock — exactly what identification of friction/damping/armature
needs. A Python control loop cannot match this (the example loop runs at 10 Hz with GC and
context-switch jitter). Under hard-RT constraints, replay is also the *easier* design to
make RT-safe than streaming, because the per-tick work becomes pure array indexing.

### 5.1 Protocol — generic timed-trajectory playback

**This is a generic xbot capability, not a sysid feature.** pyxbot and the C++ side only
know "execute this timed command trajectory and record the resulting joint states".
Everything sysid-specific — excitation design and mapping to `sysid.TimeSeries` — lives in
the adarl layer (§5.3, §6). Naming and payload stay domain-neutral so the command is
reusable for any timed trajectory playback.

The trajectory is a **time-stamped waypoint list** (it does *not* have to be at the control
rate). Each row is:

```
[ t , posref , velref , effref , stiffness , damping ]   (per commanded joint)
```

where `t` is seconds **relative to trajectory start**, and the per-joint blocks cover the
joints named in the request. `ctrl_mode` is sent **once for the whole run** (not per row).

Two non-blocking requests over the existing REQ/REP service, plus a shared
`TRAJECTORY_EXECUTION` flag (`IDLE → RUNNING → SUCCESS|ABORT → IDLE`) that the RT loop
reads:

1. **`traj_submit`** — Python sends the whole waypoint buffer + joint names + `ctrl_mode` +
   playback `mode` (`zoh` | `linear`) + caps. The **non-RT** thread copies it into a
   pre-allocated, pre-locked waypoint buffer, runs safety validation (joints exist, refs
   within limits, monotonic non-empty times, length ≤ cap), and on success sets
   `TRAJECTORY_EXECUTION = RUNNING`. Reply is immediate (accepted / rejected-with-reason);
   Python does **not** block on execution.
2. The **RT control loop** sees `RUNNING`, records the start time, and on each tick:
   - advances a **cursor** over the waypoints and computes the command for `elapsed`, by
     zero-order hold (largest `t ≤ elapsed`) or linear interpolation to the next waypoint,
     per the `mode` flag;
   - appends, for all joints, `[elapsed, pos, vel, eff]` **plus both the applied
     (post-filter) and the commanded (pre-filter) command** to the **recording buffer**, at
     the *control rate*, independent of waypoint density (see reference-filtering note).
   When the last waypoint's time passes (done) or a safety abort fires, it sets
   `TRAJECTORY_EXECUTION = SUCCESS` or `ABORT` and returns to its normal behaviour.
3. **`traj_fetch`** (poll) — once the flag is `SUCCESS`/`ABORT`, the non-RT thread
   serializes the now-static recording buffer back (raw little-endian doubles, same framing
   as the existing state message in `_extract_arrs_raw`), plus metadata: final status,
   **overrun count/flags**, **abort reason** if any, sample count, actual start time. The
   flag then returns to `IDLE`.

**While `TRAJECTORY_EXECUTION == RUNNING`, normal incoming commands are ignored** — the
command path drops them and emits a warning; the trajectory has exclusive control.

**Reference filtering.** xbot low-pass/rate-limits the references — and possibly the K/D
gains — before they reach the impedance law, so the *applied* command differs from what we
sent. The recording therefore stores **both**, per joint and per tick: the **applied
(post-filter)** `posref/velref/effref/K/D` and the **commanded (pre-filter)** ones. The
post-filter values are the physically authoritative input: for Option B (§3) they are what
must go into the impedance-law reconstruction, and logging them means the reference filter
never has to be modelled inside sysid. The pre-filter values are kept for provenance and to
validate/identify the filter itself.

`pyxbot` side: a sysid-agnostic `XbotZmqClient.execute_trajectory(waypoints, joint_names,
ctrl_mode) -> TrajectoryRecord` that submits, polls `traj_fetch`, and returns numpy arrays
of the recorded states. Turning those arrays into `sysid.TimeSeries` is the adarl sysid
layer's job, not pyxbot's.

### 5.2 RT-safety contract (the part that must be gotten right)

Everything inside the RT loop must be **O(1), allocation-free, syscall-free, lock-free**:

- **Pre-allocate + pre-lock** both buffers — the (sparse) waypoint buffer and the
  control-rate recording buffer — at `traj_submit`, before the flag goes `RUNNING`. Nothing
  is allocated once running. (xbot already `mlockall`s.)
- **Bounded size.** Waypoint count and trajectory duration are capped at submit → the
  recording buffer (`duration × control_rate`) is bounded. Budget the recording side:
  ~39 DoF × (~13 doubles: pos/vel/eff + applied & commanded refs) × 1 kHz × 30 s ≈
  **~120 MB** — still fine to pre-lock, but size the cap deliberately (drop channels or rate
  if tight).
- **Cursor, not search.** Lookup keeps a monotonically advancing index: each tick
  `while t[cursor+1] <= elapsed: cursor++`, then apply ZOH or lerp to the next waypoint per
  the `mode` flag. O(1) amortized, no unbounded scan — important under RT.
- **Flag handoff, not shared mutable buffers.** The non-RT thread owns the buffers for
  submit (before `RUNNING`) and fetch (after `SUCCESS`/`ABORT`); the RT loop owns them only
  while `RUNNING`. Writer and reader never overlap → no lock needed; transition on atomics /
  memory barriers.
- **Per-tick body** is just: advance cursor, apply waypoint, append `[elapsed,pos,vel,eff]`,
  `k++`. No I/O, no alloc.
- **Time:** `elapsed = now − start` drives both the ZOH lookup and the recorded timestamp.
  Also track per-tick deadline **overruns**; flag them and return the count so corrupted
  runs can be discarded.
- **Safety lives in the loop** (open-loop playback means Python is not guarding anything):
  in-tick clamp of references to joint/velocity/torque limits, plus an **abort condition**
  (tracking error > threshold, external e-stop, overrun) that halts playback, sets `ABORT`,
  and switches to a safe hold. The abort reason is reported in `traj_fetch` metadata.

### 5.3 Excitation design (Python, offline — produced before `sysid_load`)

- Band-limited **multisine / log-chirp** position references; small amplitudes inside
  joint limits; the rest of the joints held at a safe posture with high K.
- Sweep a few joints at a time, across several postures, for identifiability.
- Cover both **near-zero-velocity crawls** (exposes dry friction / `frictionloss`) and
  **faster sweeps** (exposes viscous `damping` + `armature`), with velocity sign reversals.
- A small `excitation.py` generates the per-joint reference arrays; this is the only part
  of collection that stays in Python.

---

## 6. Packaging recorded runs into sysid objects

With RT replay-and-record the runs arrive already uniform and time-aligned, so the former
jitter/resample handling is gone. What remains:

- **Build the TimeSeries** straight from the returned arrays: control = `eff`,
  sensordata = `pos,vel`, times = the recorded `elapsed` column (control-rate).
  `TimeSeries.from_control_names` / `from_names` map columns to model indices by joint name.
- **Initial state:** `create_initial_state(model, qpos=pos[0], qvel=vel[0],
  qpos_names=…, qvel_names=…)`. Fixed base ⇒ only actuated DoFs, no free-joint block.
- **Units / sign conventions:** confirm `JointState.eff` sign and gear vs. MuJoCo joint
  torque convention on one joint before trusting a full fit (still required).
- **Filtering:** two separate filters exist. The *reference* filter is already handled — we
  log post-filter references (§5.1), so it needs no modelling. The *joint-state*
  (measurement) low-pass should be **disabled** during capture so recorded `pos/vel/eff` are
  raw and phase-true; otherwise model its delay with `sysid.apply_delay`.
- Discard runs whose metadata reports overruns or an abort.
- Bundle each run as `ModelSequences(name, spec, sequence_name, initial_state, control,
  sensordata)`.

---

## 7. Parameters, residual, optimize

**ParameterDict** (Option A target set), per joint or shared across symmetric joints:
- `damping`   (dof_damping)
- `frictionloss` (dof_frictionloss)  ← dry friction
- `armature`  (dof_armature)         ← rotor/reflected inertia

Each as `sysid.Parameter(name, nominal, min_value, max_value, modifier=…)`. Use the
toolbox's DoF modifiers (`apply_dgain` family) or a small custom `apply_param_modifiers`
that writes `dof_damping/dof_frictionloss/dof_armature` by joint name. Start with grouped
params (e.g. one value per joint *type*) to keep it well-conditioned; un-group later.
Link inertias are intentionally **frozen** in this pass.

**Pipeline:**
```
build_model = <closure: param vector → MjSpec/MjModel via apply_param_modifiers_spec>
residual_fn = sysid.build_residual_fn(models_sequences=[...], build_model=build_model,
                                      n_threads=…, resample_true=…, sensor_weights=…)
result = sysid.optimize(initial_params, residual_fn, optimizer='mujoco'|'scipy',
                        verbose=…)
sysid.calculate_intervals(...)   # confidence intervals from the Jacobian
sysid.save_results(...)          # persist identified ParameterDict
```

`sensor_weights` to balance pos vs. vel residuals (different units/scales).

---

## 8. Validation

- **Held-out replay:** identify on run set A, predict run set B; compare predicted vs.
  measured `pos/vel` (the toolbox's `render_rollout` / `SystemTrajectory.render` for
  visual, plus RMSE per joint).
- **Confidence intervals** from `calculate_intervals` flag non-identifiable params (wide
  intervals ⇒ excitation didn't expose them ⇒ revisit §5.3).
- **Sanity:** identified `frictionloss`/`damping`/`armature` should be physically
  plausible and stable across runs.
- Bake results into the model used by [MujocoJointImpedanceAdapter](../src/adarl/adapters/MujocoJointImpedanceAdapter.py)
  (it already exposes `revolute_dof_damping_override` / `revolute_dof_frictionloss_override`
  / `safe_revolute_dof_armature`) and re-check sim-vs-real on the impedance controller.

---

## 9. Proposed module layout (when we implement)

```
xbot2_zmq/                       # C++ side (new) — generic, sysid-agnostic
  trajectory exec handler        # traj_submit/traj_fetch + TRAJECTORY_EXECUTION flag (RUNNING/SUCCESS/ABORT)
                                 # RT ZOH-replay + control-rate record, pre-alloc buffers, in-loop safety/abort

xbot2_zmq/pyxbot/src/pyxbot/
  zmq_client.py                  # add execute_trajectory(traj) -> TrajectoryRecord (sysid-agnostic)

adarl/src/adarl/sysid/           # all sysid-specific logic lives here
  centauro_model.py              # xacro→MjSpec, add torque actuators + pos/vel sensors
  excitation.py                  # multisine / chirp reference generators
  centauro_sysid_collect.py      # build excitation → execute_trajectory → sysid.TimeSeries files
  centauro_sysid_fit.py          # load runs → ParameterDict → residual → optimize → save
```

---

## 10. Open items to confirm before coding

1. `JointState.eff` — joint-side vs motor-side, and sign vs MuJoCo torque convention
   (gear ratio handling).
2. Physical fixturing for the fixed-base assumption (clamp / locked legs / hung trunk).
3. Safe excitation envelope per joint (amplitude/velocity limits, holding posture) **and**
   the in-loop abort thresholds (§5.2).
4. RT-side details: control period / max trajectory length cap, whether the existing RT
   command path can host the replay handler, and confirmation that `mlockall` + buffer
   pre-allocation cover the logging buffer.
5. Server-side joint-state filtering off during capture (§6).
6. Joint grouping for parameters (per-joint vs per-type) for the first fit.
