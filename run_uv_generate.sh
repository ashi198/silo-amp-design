#!/bin/bash

export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

uv run generate "$@"