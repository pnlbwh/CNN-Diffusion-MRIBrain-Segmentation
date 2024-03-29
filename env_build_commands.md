# Instructions for building the CNN_MASKING environment that works on modern GPUs

```bash
conda create -y -n dmri_seg python=3.9
conda activate dmri_seg
pip install tensorflow==2.11
conda install -y -c anaconda cudnn
conda install -y -c conda-forge nibabel gputil
pip install scikit-image git+https://github.com/pnlbwh/conversion.git\
pip install nvidia-pyindex nvidia-tensorrt
```

After build the environment you need to set the LD_LIBRARY_PATH to the lib of your cuda installation:

```bash
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/
```

Additionally you will need to source a compiled ANTs environment:

```bash
source /rfanfs/pnl-zorro/software/pnlpipe3/pnlpipe/soft_dir/ANTs-bin-ca32228/env.sh
```
