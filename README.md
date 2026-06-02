# ADARL - Adapters for Robot Learning

ADARL's core feature is a collection of interfaces (i.e. Adapters) that sit between Reinforcement Learning
environments and simulations or real world robotics frameworks.

The goal is to write an environment **once** against a backend-agnostic adapter interface, and then run it
unchanged on different simulators (or on real hardware) — switching backend is just a matter of swapping the
adapter. This makes it easy to, for example, train in a fast GPU-batched simulator and deploy the same
environment on a CPU simulator or a physical robot.

## Adapters

Adapters live in [`adarl/adapters`](src/adarl/adapters) and are organized as a small capability-based interface
hierarchy rather than a single monolithic API:

- **Core**: `BaseVecAdapter` provides the common lifecycle (build scenario, reset, step, read joint/link
  state, render) over a *batch* of parallel simulations.
- **Control modes** are separate mixin interfaces, so a backend only implements what it supports:
  `BaseVecJointImpedanceAdapter`, `BaseVecJointEffortAdapter`, `BaseVecJointPositionAdapter`.
- **Simulation-only** capabilities (setting state directly, spawning models, configuring collision pairs)
  live in `BaseVecSimulationAdapter`.

Concrete backends:

| Adapter | Backend | Notes |
|---|---|---|
| `MjxAdapter`, `MjxJointImpedanceAdapter`, `MjxActuatedAdapter` | MuJoCo **MJX** | GPU-batched, JAX/XLA/Warp |
| `MujocoAdapter`, `MujocoJointImpedanceAdapter` | MuJoCo **classic** | CPU, single simulation (`vec_size == 1`) |
| `PyBulletAdapter`, `PyBulletJointImpedanceAdapter`, `PyBullet2DofCartesianAdapter` | PyBullet | CPU |
| `ZmqXbotAdapter` / `VecZmqXbotAdapter`, `StandaloneRealAdapter` | Real robots | hardware deployment (e.g. XBot over ZMQ) |

`VecSimJointImpedanceAdapterWrapper` adapts a single-simulation adapter to the vectorized (`Vec`) interface,
so single-sim backends can be used where a batched adapter is expected.

## Environment structure

Environments are **vectorized**: they operate on a batch of robots at once (a batch of size 1 is just the
degenerate case), which is what lets a single environment scale from one CPU simulation up to thousands of
GPU-batched ones. The framework lives in [`adarl/envs/vec`](src/adarl/envs/vec) and follows a runner-based
design:

- **Env logic** — `BaseVecEnv` → `ControlledVecEnv` define how observations, rewards and resets are computed
  for the batch, driving the simulation through an adapter.
- **Runner** — `EnvRunner` drives the env: stepping, reset / autoreset, episode bookkeeping and logging.
  Wrappers like `EnvRunnerRecorderWrapper` add video/info recording on top of a runner.
- **Gym exposure** — `Runner2GymWrapper` and `Runner2VecGymWrapper` expose a runner as a standard
  Gymnasium env / vector env, so trained agents and existing RL tooling can consume it directly.

(A set of older single-environment classes also exists under `adarl/envs`, but are kept only for legacy reasons.)

## Repository layout

- [`adarl/adapters`](src/adarl/adapters) — the adapter interfaces and backend implementations.
- [`adarl/envs/vec`](src/adarl/envs/vec) — the vectorized environment framework (env base classes, runner, Gym wrappers).
- [`adarl/utils`](src/adarl/utils) — supporting utilities (logging, tensor-tree helpers, run/session management, debugging).
- [`adarl/models`](src/adarl/models), [`adarl/assets`](src/adarl/assets) — robot/scene models and assets.
- [`adarl/examples`](src/adarl/examples) — runnable examples (e.g. cartpole, ant, half-cheetah).
