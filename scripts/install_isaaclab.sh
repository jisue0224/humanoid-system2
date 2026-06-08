#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3.10 -m venv env_isaaclab
source env_isaaclab/bin/activate

python -m pip install --upgrade pip

# Some legacy Isaac Lab dependencies still import pkg_resources while building.
python -m pip install "setuptools<80" wheel

# Isaac Lab 2.1 docs target CUDA 12 PyTorch 2.5.1 wheels for Python 3.10.
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Avoid build isolation pulling a setuptools release without pkg_resources.
python -m pip install flatdict==4.0.1 --no-build-isolation

# Isaac Lab pip packages include Isaac Sim. This path is recommended for external runner scripts.
python -m pip install "isaaclab[isaacsim,all]==2.1.0" --extra-index-url https://pypi.nvidia.com

mkdir -p external
if [ ! -d external/IsaacLab/.git ]; then
  git clone --depth 1 --branch v2.1.0 https://github.com/isaac-sim/IsaacLab.git external/IsaacLab
fi

# The pip package can install Isaac Sim and core Isaac Lab, but source install gives us
# bundled task registries and runner scripts needed for H1 experiments.
# IsaacLab v2.1.0's installer always installs source extensions; the argument
# selects the RL framework extra.
TERM=xterm external/IsaacLab/isaaclab.sh --install rsl_rl

python scripts/check_system.py
