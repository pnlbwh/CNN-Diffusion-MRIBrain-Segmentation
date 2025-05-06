![](pnl-bwh-hms.png)

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.3665739.svg)](https://doi.org/10.5281/zenodo.3665739) [![Python](https://img.shields.io/badge/Python-3.6-green.svg)]() [![Platform](https://img.shields.io/badge/Platform-linux--64%20%7C%20osx--64-orange.svg)]()

*CNN-Diffusion-MRIBrain-Segmentation* repository is developed by Senthil Palanivelu, Suheyla Cetin Karayumak, Tashrif Billah, Ryan Zurrin, Sylvain Bouix, and Yogesh Rathi, 
Brigham and Women's Hospital (Harvard Medical School).

Table of Contents
=================

* [Table of Contents](#table-of-contents)
* [Segmenting diffusion MRI brain](#segmenting-diffusion-mri-brain)
* [Citation](#citation)
* [Dependencies](#dependencies)
* [Installation](#installation)
   * [1. Python 3](#1-python-3)
   * [2. Conda environment](#2-conda-environment)
   * [3. ANTs](#3-ants)
   * [4. CUDA environment](#4-cuda-environment)
   * [5. Download models](#5-download-models)
   * [6. Run script](#6-run-script)
   * [7. mrtrix](#7-mrtrix)
   * [8. FSL](#8-fsl)
* [Singularity container](#singularity-container)
* [Running the pipeline](#running-the-pipeline)
   * [Prediction](#prediction)
   * [Training](#training)
      * [1. B0 extraction](#1-b0-extraction)
      * [2. Registration](#2-registration)
      * [3. Preprocessing](#3-preprocessing)
      * [4. Deep learning](#4-deep-learning)
* [Method](#method)
   * [1. Model Architecture](#1-model-architecture)
   * [2. Multi View Aggregation:](#2-multi-view-aggregation)
   * [3. Clean up](#3-clean-up)
* [Troubleshooting](#troubleshooting)
* [Issues](#issues)
* [Reference](#reference)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->


# Segmenting diffusion MRI brain

A challenging problem in neuroscience is to create brain mask of an MRI. The problem becomes more challenging when 
the MRI is a diffusion MRI (dMRI) since it has less contrast between brain and non-brain regions. Researchers all over the 
world have to spent many hours drawing dMRI brain masks manually. In order to automate that process, we come up with a 
convolutional neural network (CNN) based approach for creating dMRI brain mask automatically. The CNN architecture is 
obtained from Raunak Dey's [work](https://github.com/raun1/MICCAI2018---Complementary_Segmentation_Network-Raw-Code). 
On top of that, we applied a multi-view aggregation step to obtain final brain mask.

The software is capable of both training and predicton. We also provide trained models so it can be used off the shelf.
Psychiatry NeuroImaging Laboratory has a large number of dMRI brain masks drawn by research assistants. 
The models were trained on 1,500 such brain masks. We acchieved an accuracy of 97% in properly segmenting the brain 
from a dMRI b0 volume.



# Citation

If you use this software, please cite all of the following:

* Palanivelu, S., Cetin Karayumak, S., Billah, T., Zurrin, R., Bouix, S., Rathi, Y.; CNN based diffusion MRI brain segmentation tool, 
https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation, 2020, DOI: 10.5281/zenodo.3665739

* Cetin-Karayumak, S., Zhang, F., Zurrin, R. et al. Harmonized diffusion MRI data and white matter measures from the
Adolescent Brain Cognitive Development Study. Sci Data 11, 249 (2024). https://doi.org/10.1038/s41597-024-03058-w


# Dependencies

* Python 3
* ANTs

# Installation

## 1. Python 3

Download [Miniconda Python 3 bash installer](https://docs.conda.io/en/latest/miniconda.html) (32/64-bit based on your environment):
    
    sh Miniconda3-latest-Linux-x86_64.sh -b # -b flag is for license agreement

Activate the conda environment:

    source ~/miniconda3/bin/activate # should introduce '(base)' in front of each line

The software is written intelligently to work with or without GPU support.

<details><summary>Old conda environment</summary>

<br>
  
We had provided two [*yml* files](../archive/) to facilitate the creation of `dmri_seg` conda environment.

* In the initial stage of this program development in 2020, the [CPU environment](../archive/environment_cpu.yml) built and worked fine.
  But it does not work anymore.
* The [GPU environment](../archive/environment_gpu.yml) works fine on older GPUs.
  But in 2023, it did not produce a valid mask on modern GPUs e.g. NVIDIA RTX 4000, A6000, A100.

</details>

In 2024, we recommend the below step-by-step instructions to build an environment that
  successfully generated valid masks on both GPU and CPU devices.
   
## 2. Conda environment

Step-by-step instructions:

```
conda create -y -n dmri_seg python=3.11 -c conda-forge --override-channels
conda activate dmri_seg
pip install 'tensorflow[and-cuda]==2.15.1'
pip install scikit-image gputil git+https://github.com/pnlbwh/conversion.git
```

Tensorflow installation is inspired by https://www.tensorflow.org/install/pip.

## 3. ANTs

You need to have a working `antsRegistrationSyNQuick.sh` in your `PATH`.
You can build ANTs following their [GitHub](https://github.com/ANTsX/ANTs).
Or you can use our pre-built ANTs:

> conda install -y pnlbwh::ants

Either way, you need to set the environment variable `ANTSPATH`. For the latter method, it is:

> export ANTSPATH=${CONDA_PREFIX}/bin

## 4. CUDA environment

To run this program on GPU, you must set `LD_LIBRARY_PATH`:

> export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/:${LD_LIBRARY_PATH}

Your NVIDIA driver should be compatible with CUDA.

<details><summary>System's CUDA</summary>
<br>
  
In some servers, a matched CUDA may be available in `/usr/local/cuda-*/`
directory. You can also use that CUDA instead of `${CONDA_PREFIX}/lib/`:

    # add cuda tools to your PATH
    export PATH=/usr/local/cuda-9.1/bin:${PATH}

    # add cuda libraries to your LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64:${LD_LIBRARY_PATH}

</details>

## 5. Download models

Download model architecture, weights and IIT mean b0 template from 
https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation/releases as follows:

    cd CNN-Diffusion-MRIBrain-Segmentation
    wget https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation/releases/download/v0.3/model_folder.tar.gz
    tar -xzvf model_folder.tar.gz
    cd model_folder
    wget https://www.nitrc.org/frs/download.php/11290/IITmean_b0_256.nii.gz


They will be extracted to `CNN-Diffusion-MRIBrain-Segmentation/model_folder` directory.


## 6. Run script

To avoid the need for setting `LD_LIBRARY_PATH` and providing Python interpreter every time `dwi_masking.py` is run,
we wrote a run script `dwi_masking.sh`. If you would like, you can configure it for your environment as:

```
cd CNN-Diffusion-MRIBrain-Segmentation/pipeline
sed -i "s+PREFIX+${CONDA_PREFIX}+g" dwi_masking.sh
```

Afterward, you can use `dwi_masking.sh` instead of `dwi_masking.py` in all the examples shown in this manual.


## 7. mrtrix

This software makes use of only one binary from mrtrix, `maskfilter`. If you already have mrtrix installed on your 
system, you can use it. Moreover, you can follow [this](https://mrtrix.readthedocs.io/en/latest/installation/linux_install.html) instruction to install mrtrix.
However, if you are unable to use/install mrtrix, just set `-filter scipy` argument and the software will 
use a Python translated version of the above binary.

See [Clean up](#3-clean-up) for details.


## 8. FSL

This software marginally depends on FSL in the sense that you need FSL's `slicesdir` executable
to generate snapshots of image-mask for being able to QC it. Hence, we do not recommend
sourcing the entire FSL environment. Instead, you should just put `slicesdir` in your PATH:

> export PATH=/path/to/fsl-6.0.7/share/fsl/bin:$PATH

In additon, you should set:

```
export FSLDIR=/path/to/fsl-6.0.7/
export FSLOUTPUTTYPE=NIFTI_GZ
```


# Singularity container

The dependencies in [environment_cpu.yml](../environment_cpu.yml) have been found to be volatile. So we have archived 
the dependencies of this software in a [Singularity](../Singularity) container. After you have downloaded the models, 
you can download and use the it as follows:

    # download the container
    cd CNN-Diffusion-MRIBrain-Segmentation
    wget https://www.dropbox.com/s/rpc8ighcxqvzete/dmri_seg.sif
    
    # verify that it works
    singularity run dmri_seg.sif dwi_masking.py --help
    
    # use it to generate brain masks
    singularity run \
    --bind /path/to/data:/data \
    --bind model_folder:/home/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation/model_folder \
    dmri_seg.sif \
    dwi_masking.py -i /data/dwi_list.txt -f /home/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation/model_folder


Please continue reading below to learn more about `dwi_list.txt`. Also note that, the container supports only `-filter scipy` option.
    

# Running the pipeline


## Prediction

Prediction refers to creation of masks based on pre-trained model. This is the common use of this software.
    
    pipeline/dwi_masking.py -i dwi_list.txt -f model_folder
    pipeline/dwi_masking.py -i dwi_list.txt -f model_folder -nproc 16
    pipeline/dwi_masking.py -i b0_list.txt -f model_folder
    pipeline/dwi_masking.py -i b0_list.txt -f model_folder -qc 1


* `dwi_list.txt` and `b0_list.txt` must contain full paths to diffusion and b0 volumes respectively

* Each created mask is saved in the directory of input volume with name `dwib0_{PREFIX}-multi_BrainMask.nii.gz`


## Training

However, you can train a model on your own data using this software. 
Data refers to a set of b0 images and their corresponding masks.
Training is performed in four steps:

### 1. B0 extraction

In the training stage, we train a CNN to predict b0 brain mask with reasonable accuracy. For the CNN to learn masking, 
we have to provide a set of b0 images and their corresponding masks. So the first step in training would extracting 
b0 image given a DWI. You can use your favorite tool, bvalue threshold, and method (first b0 or average over all b0s) 
to obtain b0 image. Then, researchers usually draw masks by hand or edit FSL `bet` produced masks. 
Once you have a set of b0s and their corresponding masks, you can start the next steps.


### 2. Registration

This software uses `IITmean_b0_256.nii.gz` image as the reference that you can download 
from https://www.nitrc.org/frs/?group_id=432 . If your b0s and corresponding masks are already in the space of 
`IITmean_b0_256.nii.gz` (ICBM-152/MNI152), you do not need to register again.

Example:

    src/registration.py -b0 b0_list.txt -mask mask_list.txt -ref model_folder/IITmean_b0_256.nii.gz
    

The registered images are stored in the directory of given images as `{Prefix}-Warped.nii.gz` and 
`{Prefix}-Warped-mask.nii.gz`.

### 3. Preprocessing

Whether you register the images or not, create a text file for with their absolute paths:

> b0_list.txt

    /path/to/case1_b0.nii.gz
    /path/to/case2_b0.nii.gz
    ...
    

> mask_list.txt

    /path/to/case1_mask.nii.gz
    /path/to/case2_mask.nii.gz
    ...
    
Make sure to account for registered paths if have have done so.

Then, we normalize b0 image to the range [0,1] with respect to its highest intensity. 
Same clipping is performed to the mask. 

    src/preprocess_b0.py -i b0_list.txt
    src/preprocess_mask.py -i mask_list.txt

Upon preprocessing, b0 data are appended to:
    
    sagittal-traindata-dwi.npy
    coronal-traindata-dwi.npy
    axial-traindata-dwi.npy

Similarly, mask data are appended to files with `-mask.npy` suffix. The above `*.npy` files are saved in 
the directory of the first b0 image given in `b0_list.txt`.


### 4. Deep learning

Now we come to the actual machine learning stage. We want to train our CNN varying `principal_axis` 
over the set `{axial,coronal,saggittal}` one at a time. 

Copy the `src/settings.ini` to a different directory and edit relevant fields:

```ini
[COMMON]
save_model_dir = "project/model_dir/"
log_dir = "project/log_dir"

[DATA]
data_dir = "project/dir/of/first/case/"
train_data_file = "sagittal-traindata-dwi.npy"
train_label_file = "sagittal-traindata-mask.npy"

[TRAINING]
principal_axis = "sagittal"
learning_rate = 1e-3
train_batch_size = 4
validation_split = 0.2
num_epochs = 1
shuffle_data = "True"
```

At the very least, you should edit `save_model_dir`, `log_dir`, `data_dir`, and `num_epochs` fields. Now define 
environment variable `COMPNET_CONFIG` so your own `settings.ini` could be found by `src/train.py` program:

    export COMPNET_CONFIG=/path/to/your/settings.ini
    
    
Finally, you can use the same `settings.ini` file for training three models corresponding to three 
principal axes- `sagittal`, `coronal`, and `axial`. For each of the axes, run:

    src/train.py  # for sagittal, num_epochs=9
    src/train.py  # for coronal, num_epochs=8
    src/train.py  # for axial, num_epochs=8


**NOTE** The above `num_epochs` were found suitable for the models distributed with this software.

`src/train.py` will save each epoch `*-compnet_final_weight-*.h5` models in `save_model_dir` that you defined in `settings.ini`.
You can use this `save_model_dir` for [prediction](#prediction). Note that, the final epoch models are used in prediction by default. 
If you want to use another epoch models other than the final ones created above, save them in a different directory along with 
`CompNetBasicModel.json` and `IITmean_b0_256.nii.gz` and use that directory as `save_model_dir` for prediction. As you probably 
realized already, the three models pertinent to three views would not have to be of the same epoch number.


# Method

## 1. Model Architecture

The code is written by Raunak Dey available at 

https://github.com/raun1/MICCAI2018---Complementary_Segmentation_Network-Raw-Code. 

In summary, his proposed architecture is designed in the framework of encoder-decoder networks and have three pathways.

* Segmentation Branch - learns what is the brain tissue and to generate a brain mask 

* Complementary Branch - learns what is outside of the brain and to help the other
branch generate brain mask

* Reconstruction Branch - It provides direct feedback to the segmentation and
complementary branche and expects reasonable predictions from them as input to reconstruct the original input image.

![](CompNet.png)
**Illustration kindly provided by Raunak Dey**


## 2. Multi View Aggregation:

The approach is to train three separate networks for three principal axes ( Sagittal, Coronal and axial ) and 
to perform multi-view aggregation step that combines segmentations from models trained on 2D slices along three principal axes: 
coronal, sagittal and axial. The final segmentation is obtained by combining the probability maps from all three segmentation.

![](Multiview.png)


See [Reference #2](#reference) for details of this method.


## 3. Clean up

The aggregated mask can be cleaned up using a [maskfilter from mrtrix](https://mrtrix.readthedocs.io/en/latest/reference/commands/maskfilter.html) by `-filter mrtrix` argument. 
In brief, there remain islands of non brain region in the aggregated mask. The above filter applies a series of morphological 
operation i.e. erosion and dilation to clean up the mask. If mrtrix installation is a problem for you, 
you can use [Python translated version](../src/maskfilter.py) of maskfilter provided with this software by `-filter scipy` argument. If `-filter` is not used, then the aggregated mask is returned.


# Troubleshooting

1. If you get any error while building `dmri_seg` conda environment, updating conda first and retrying should be helpful:

```    
conda update -y -n base -c defaults conda
```

See https://github.com/conda/conda/issues/8661 for details.

# Issues

If you experience any issue with using this software, please open an issue [here](https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation/issues).
We shall get back to you as soon as we can.


# Reference

1. Dey, Raunak and Hong, Yi. "CompNet: Complementary segmentation network for brain MRI extraction." 
International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2018.

2. Roy, Abhijit Guha, et al. "QuickNAT: A fully convolutional network for quick and accurate segmentation of neuroanatomy." 
NeuroImage 186 (2019): 713-727.

3. Billah, Tashrif; Bouix, Sylvain; Rathi, Yogesh; Various MRI Conversion Tools, https://github.com/pnlbwh/conversion, 2019, 
DOI: 10.5281/zenodo.2584003

4. Zhang S, Arfanakis K. "Evaluation of standardized and study-specific diffusion tensor imaging templates of 
the adult human brain: Template characteristics, spatial normalization accuracy, and detection of small 
inter-group FA differences." Neuroimage 2018;172:40-50.

