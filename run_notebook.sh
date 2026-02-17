#!/bin/bash
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=-1
source .venv/bin/activate
jupyter nbconvert --to notebook all-python.ipynb --execute "$@"
