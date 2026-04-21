#!/bin/bash
source ~/.bashrc
source examples/libero/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD/third_party/libero"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0