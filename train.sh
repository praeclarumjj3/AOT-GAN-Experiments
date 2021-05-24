#!/bin/sh

python3 src/train.py --data_train 'COCOA' --subset_factor 1 --rec_loss '1*L1+0.1*Perceptual' --model_name 'aot_smPatchGAN'
rm -rf `find -type d -name .ipynb_checkpoints`