#!/bin/bash

uv pip install -r src/adarl/requirements_py313.txt
uv pip install -e src/adarl -e src/adarl_envs -e src/rreal  -e src/pykyon  -e src/pycentauro  -e src/pydagana