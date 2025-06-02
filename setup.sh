#!/bin/bash

#Exit script on error
set -e

# System dependencies (WSL or Ubuntu)
sudo apt update && sudo apt install -y \
  build-essential \
  swig \
  python3-dev \
  python3-pip \
  python3-setuptools \
  python3-wheel \
  cmake \
  libatlas-base-dev \
  libopenblas-dev \
  liblapack-dev \
  libffi-dev \
  libxml2-dev \
  libgl1 \
  libgl1-mesa-glx \
  default-jre \
  curl

#create the virtual environmnet if not present
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip + build tools before installing anything else
pip install --upgrade pip setuptools wheel

# Pre-install numpy and cython to avoid build issues with auto-sklearn
pip install numpy==1.24.4 cython==0.29.36 scikit-learn==1.2.2

pip install --no-cache-dir https://h2o-release.s3.amazonaws.com/h2o/rel-3.46.0/7/Python/h2o-3.46.0.7-py2.py3-none-any.whl

# Install rest of the Python dependencies
pip install --no-cache-dir -r requirements.txt
