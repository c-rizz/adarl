"""Root pytest configuration for the adarl test suite.

Layout
------
    tests/utils/      pure-python / torch unit tests (fast, CPU, no simulator)
    tests/adapters/   simulation-adapter tests (mujoco = CPU; mjx/genesis = GPU)

    tests/adapters/adapter_compliance.py  shared, backend-agnostic behavioral checks
    tests/adapters/_backends.py           backend capability detection + adapter builders

Slicing the suite (markers are declared in pyproject.toml)::

    pytest -m "not gpu"     # fast CPU-only tests (unit + mujoco)
    pytest -m gpu           # only the GPU-backed adapter tests
    pytest tests/utils      # just the unit tests

Tests that need hardware or an optional backend skip themselves when the requirement is
missing (no GPU, backend not installed), so the suite is always runnable everywhere.
"""
