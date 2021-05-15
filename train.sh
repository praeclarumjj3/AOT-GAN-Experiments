#!/bin/sh

python3 src/train.py
rm -rf `find -type d -name .ipynb_checkpoints`