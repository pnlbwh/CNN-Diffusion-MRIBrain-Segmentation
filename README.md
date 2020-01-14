# CompNet - Keras Implementation

## Tool:CompNet: Segmenting diffusion brain MRI

The code for training, as well as the Trained Models are provided here.
The model is trained on 1500 b0 diffusion volumes.

> Model Architecture: https://drive.google.com/open?id=163KTt2ilmz1RqUgXcWu6IAH1DLO3gOoI

> Trained Model Weights: https://drive.google.com/open?id=111x4xYxzDpUxlgNV83llpQdI3CxMVMdd

Let us know if you face any problems running the code by posting in Issues.

If you use this code please cite:

Raunak Dey, Yi Hong, C.2018 CompNet: Complementary Segmentation Network for Brain MRI Extraction . Accepted to MICCAI 2018 https://arxiv.org/abs/1804.00521

Guha Roy, A., Conjeti, S., Navab, N., and Wachinger, C. 2018. QuickNAT: A Fully Convolutional Network for Quick and Accurate Segmentation of Neuroanatomy. Accepted for publication at NeuroImage, https://arxiv.org/abs/1801.04161

## Getting Started

### Pre-requisites

You need to have following in order for this library to work as expected

01)  python 2.7
02)  pip >= 19.0
03)  numpy >= 1.14.0
04)  nibabel >= 2.2.1
05)  nilearn >= 0.5.0
06)  opencv-python >= 3.4.1.15
07)  pandas >= 0.23.0
08)  scikit-image >= 0.13.1
09)  scikit-learn >= 0.20.0
10)  tensorflow-gpu >= 1.12.0
11)  keras >= 2.1.6
12)  cudatoolkit = 9.0
13)  cudnn = 7.0.5

### Code Author
* Raunak Dey - [raun1](https://github.com/raun1)
### Model Architecture
The proposed networks are designed in the framework of encoder-decoder networks and have three pathways.
> 1) Segmentation Branch - learns what is the brain tissue and to generate a brain mask 
> 2) Complementary Branch - learns what is outside of the brain and to help the other
branch generate brain mask
> 3) Reconstruction Branch - It provides direct feedback to the segmentation and
complementary branche and expects reasonable predictions from them as input to reconstruct the original input image.
![Screenshot](https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation/blob/master/CompNet%20Arch.png)


### Multi View Aggregation step:
> The approach is to train 3 separate networks for three principal axes ( Sagittal, Coronal and axial ) and 
to perform multi-view aggregation step that combines segmentations from models trained on 2D slices along three principal axes: coronal, sagittal and axial. The final segmentation would be obtained by combining the probability maps from all three segmentation.
![Screenshot](https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation/blob/master/Multiview.png)

