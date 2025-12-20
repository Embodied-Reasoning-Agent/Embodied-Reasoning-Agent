#!/bin/bash
set -e

echo "Starting installation process..."

pip install 'qwen-vl-utils'
pip install 'mathruler'
pip install 'matplotlib'
pip install 'flask'

echo "Installing flash-attn with no build isolation..."
pip install flash-attn==2.7.4.post1

echo "Installing vagen package..."
pip install -e .

pip install 'gym'
pip install 'gymnasium'

echo "Installation complete"
