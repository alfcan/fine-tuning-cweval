#!/bin/bash
# setup_env.sh - Setup script for the Preference-based Security Alignment project

echo "=== Phase 0: Environment Setup ==="

# 1. Clone CWEval repo if not already cloned
if [ ! -d "CWEval" ]; then
    echo "Cloning CWEval repository..."
    git clone https://github.com/Co1lin/CWEval.git
else
    echo "CWEval repository already exists."
fi

# 2. Pull official CWEval Docker image
echo "Pulling the official CWEval Docker image..."
docker pull co1lin/cweval

# 3. Create virtual environment if requested, or install packages
echo "Installing required Python packages..."
pip install --upgrade pip
pip install \
    transformers \
    peft \
    trl \
    accelerate \
    bitsandbytes \
    litellm \
    pandas \
    numpy \
    matplotlib \
    scipy \
    requests \
    tqdm

echo "=== Setup Complete ==="
echo "To run CWEval tests inside Docker container:"
echo "  docker run --name cweval --rm -it --net=host -v \$(pwd):/workspace co1lin/cweval zsh"
