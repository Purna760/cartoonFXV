#!/usr/bin/env bash
set -o errexit

pip install --upgrade pip
pip install -r render_requirements.txt

mkdir -p static/uploads static/processed
