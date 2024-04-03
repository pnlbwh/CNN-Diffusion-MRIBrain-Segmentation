#!/bin/bash

# Before running this script:
# source /path/to/miniconda3/bin/activate

conda create -y -n dmri-seg python=3.9
conda activate dmri-seg
pip install tensorflow==2.11
conda install -y anaconda::cudnn conda-forge::gputil
pip install nibabel scikit-image git+https://github.com/pnlbwh/conversion.git

