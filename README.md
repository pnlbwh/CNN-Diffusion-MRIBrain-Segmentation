# CompNet - Keras Implementation

## Tool:CompNet: Segmenting diffusion brain MRI

The code for training, as well as the Trained Models are provided here.
The model is trained on 1500 b0 diffusion volumes.

Let us know if you face any problems running the code by posting in Issues.

If you use this code please cite:

Raunak Dey, Yi Hong, C.2018 CompNet: Complementary Segmentation Network for Brain MRI Extraction . Accepted to MICCAI 2018 https://arxiv.org/abs/1804.00521

Guha Roy, A., Conjeti, S., Navab, N., and Wachinger, C. 2018. QuickNAT: A Fully Convolutional Network for Quick and Accurate Segmentation of Neuroanatomy. Accepted for publication at NeuroImage, https://arxiv.org/abs/1801.04161

## Getting Started

### Pre-requisites

You need to have following in order for this library to work as expected

01)  python 3.6
02)  pip >= 19.0
03)  numpy >= 1.16.4
04)  nibabel >= 2.2.1
05)  tensorflow-gpu >= 1.12.0
06)  keras >= 2.2.4
07)  cudatoolkit = 9.0
08)  cudnn = 7.0.5

### Python 3

Download [Miniconda Python 3.6 bash installer](https://docs.conda.io/en/latest/miniconda.html) (32/64-bit based on your environment):
    
    sh Miniconda3-latest-Linux-x86_64.sh -b # -b flag is for license agreement

Activate the conda environment:

    source ~/miniconda3/bin/activate # should introduce '(base)' in front of each line
    
### Install prerequisites for running the pipeline

01) conda install cudatoolkit=9.0
02) conda install cudnn=7.0.5
03) conda install -c pnlbwh ants
04) pip install tensorflow==1.12.0
05) pip install tensorflow-gpu==1.12.0
06) pip install keras==2.2.4
07) pip install nibabel
18) pip install gputil

### Setting CUDA Path
The NVIDIA graphics driver and CUDA compilier are already installed on machines that support CUDA. However, one must set environment variables in order to run and write CUDA enabled programs.

If you use bash, add the following lines to the bottom of your .bashrc file:

        # add cuda tools to command path
        export PATH=/usr/local/cuda/bin:${PATH}

        # add the CUDA binary and library directory to your LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
  
Log out and back in for the changes to take effect.

### Download model architecture, weights and IIT mean b0 template

Download the following data and place them under `model_folder/` directory
> Model Architecture: https://drive.google.com/open?id=163KTt2ilmz1RqUgXcWu6IAH1DLO3gOoI

> Trained Model Weights: https://drive.google.com/open?id=111x4xYxzDpUxlgNV83llpQdI3CxMVMdd

> Reference b0 Image: https://drive.google.com/open?id=1Mc8ZXCguRNl67wxY7z8EM9SMXGnt7VEc

### Running the pipeline

##### Step1
```
python preprocessing.py -i subject/cases.txt -ref model_folder/IITmean_b0_256.nii.gz
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Note: cases.txt should contain the full path to the diffusion volumes
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/home/pycharm/data/compnet/subject01/subject01_dwi.nii.gz
##### Step 2
```
python dwi_masking.py -i subject/cases.txt -f model_folder/
```
##### Step 3
```
python postprocessing.py -i subject/cases.txt
```

## Code Author
* Raunak Dey - [raun1](https://github.com/raun1)
## Model Architecture
The proposed networks are designed in the framework of encoder-decoder networks and have three pathways.
> 1) Segmentation Branch - learns what is the brain tissue and to generate a brain mask 
> 2) Complementary Branch - learns what is outside of the brain and to help the other
branch generate brain mask
> 3) Reconstruction Branch - It provides direct feedback to the segmentation and
complementary branche and expects reasonable predictions from them as input to reconstruct the original input image.
![Screenshot](https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation/blob/master/CompNet%20Arch.png)


## Multi View Aggregation step:
> The approach is to train 3 separate networks for three principal axes ( Sagittal, Coronal and axial ) and 
to perform multi-view aggregation step that combines segmentations from models trained on 2D slices along three principal axes: coronal, sagittal and axial. The final segmentation would be obtained by combining the probability maps from all three segmentation.
![Screenshot](https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation/blob/master/Multiview.png)

