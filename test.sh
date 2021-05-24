#!/bin/sh

python src/test.py --pre_train 'experiments/G0000000.pt' --data_test 'COCOA'
rm -rf `find -type d -name .ipynb_checkpoints`