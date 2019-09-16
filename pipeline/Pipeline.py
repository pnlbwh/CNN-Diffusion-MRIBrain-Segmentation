#!/rfanfs/pnl-zorro/home/suheyla/Tools/tensorflow-prebuiltin-pycharm/bin/python
from __future__ import division
# -----------------------------------------------------------------
# Author:		PNL BWH                 
# Written:		07/02/2019                             
# Last Updated: 	09/13/2019
# Purpose:  		Python pipeline for diffusion brain masking
# -----------------------------------------------------------------

"""
CompNet.py
~~~~~~~~~~
1)  Accepts the diffusion image in *.nhdr,*.nrrd,*.nii.gz,*.nii format
2)  Checks if the Image axis is in the correct order for *.nhdr and *.nrrd file
3)  Extracts b0 Image
4)  Converts nhdr to nii.gz
5)  Re sample nii.gz file to 246 x 246
6)  Pads the Image adding zeros to 256 x 256
7)  Normalize the Image by 99th percentile
8)  Applys Rigid-Body tranformation to standard MNI space using
9)  Neural network brain mask prediction across the 3 principal axis
10) Performs Multi View Aggregation
11) Converts npy to nhdr,nrrd,nii,nii.gz
12) Applys Inverse tranformation
13) Down sample to original resolution
14) Peforms Binary Dilation
15) Cleaning
"""


# pylint: disable=invalid-name
import os
import os.path
from os import path
import webbrowser
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress tensor flow message
import GPUtil 

# Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Get the first available GPU
try:
    DEVICE_ID_LIST = GPUtil.getFirstAvailable()
    DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list
    print("GPU found...", DEVICE_ID)

    # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

except RuntimeError:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("GPU not available...")

import tensorflow as tf
import re
import sys
import subprocess
import argparse, textwrap
import datetime
import os.path
import pathlib
import nibabel as nib
import numpy as np
import multiprocessing as mp
import scipy.ndimage as nd
from os import path
from keras.models import load_model
from keras.models import model_from_json
import cv2
import sys
import keras
import scipy as sp
import scipy.misc, scipy.ndimage.interpolation
import os
from keras import losses
from keras.models import Model
from keras.layers import Input, merge, concatenate, Conv2D, MaxPooling2D, \
    Activation, UpSampling2D, Dropout, Conv2DTranspose, add, multiply
from keras.layers.normalization import BatchNormalization as bn
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from keras import regularizers
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# suffixes
SUFFIX_NIFTI = "nii"
SUFFIX_NIFTI_GZ = "nii.gz"
SUFFIX_NRRD = "nrrd"
SUFFIX_NHDR = "nhdr"
SUFFIX_NPY = "npy"
SUFFIX_TXT = "txt"


def predict_mask(input_file, view='default'):
    """
    Parameters
    ----------
    input_file : str
                 (single case filename which is stored in disk in *.nii.gz format) or 
                 (list of cases, all appended to 3d numpy array stored in disk in *.npy format)
    view       : str
                 Three principal axes ( Sagittal, Coronal and Axial )
    
    Returns
    -------
    output_file : str
                  returns the neural network predicted filename which is stored
                  in disk in 3d numpy array *.npy format
    """
    print "Loading " + view + " model from disk..."
    smooth = 1.

    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    # Negative dice to obtain region of interest (ROI-Branch loss) 
    def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)

    # Positive dice to minimize overlap with region of interest (Complementary branch (CO) loss)
    def neg_dice_coef_loss(y_true, y_pred):
        return dice_coef(y_true, y_pred)

    # load json and create model
    json_file = open('/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/model/CompNetmodel_arch_DWI_percentile_99.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(
        '/rfanfs/pnl-zorro/home/sq566/pycharm/Suheyla/model/' + view + '.h5')

    # evaluate loaded model on test data
    loaded_model.compile(optimizer=Adam(lr=1e-5),
                         loss={'output1': dice_coef_loss, 'output2': dice_coef_loss, 'output3': dice_coef_loss,
                               'output4': dice_coef_loss, 'conv10': dice_coef_loss, 'final_op': dice_coef_loss,
                               'xoutput1': neg_dice_coef_loss, 'xoutput2': neg_dice_coef_loss,
                               'xoutput3': neg_dice_coef_loss,
                               'xoutput4': neg_dice_coef_loss, 'xconv10': neg_dice_coef_loss,
                               'xfinal_op': neg_dice_coef_loss,
                               'xxoutput1': 'mse', 'xxoutput2': 'mse', 'xxoutput3': 'mse', 'xxoutput4': 'mse',
                               'xxconv10': 'mse', 'xxfinal_op': 'mse'})

    case_name = os.path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-' + view + '-mask.npy'
    output_file = os.path.join(os.path.dirname(input_file), output_name)

    if input_file.endswith(SUFFIX_NIFTI_GZ):
        x_test = nib.load(input_file).get_data()
        if view == 'coronal':
            x_test = np.swapaxes(x_test, 0, 1)  # sagittal to coronal view
        elif view == 'axial':
            x_test = np.swapaxes(x_test, 0, 2)  # sagittal to axial view
    else:
        x_test = np.load(input_file)

    x_test = x_test.reshape(x_test.shape + (1,))
    predict_x = loaded_model.predict(x_test, verbose=1)
    SO = predict_x[5]  # Segmentation Output
    del predict_x

    if input_file.endswith(SUFFIX_NIFTI_GZ):
        if view == 'coronal':
            SO = np.swapaxes(SO, 1, 0) # coronal to sagittal view
        elif view == 'axial':
            SO = np.swapaxes(SO, 2, 0) # axial to sagittal view
    np.save(output_file, SO)
    return output_file


def multi_view_agg(sagittal_SO, coronal_SO, axial_SO, input_file):
    """
    Parameters
    ----------
       sagittal_SO : str
                     Sagittal view predicted mask filename which is in 3d numpy *.npy format stored in disk
       coronal_SO  : str
                     coronal view predicted mask filename which is in 3d numpy *.npy format stored in disk
       axial_SO    : str
                     axial view predicted mask filename which is in 3d numpy *.npy format stored in disk
       input_file  : str
                     single input case filename which is in *.nhdr format
    Returns
    -------
       output_file : str
                     Segmentation file name obtained by combining the probability maps from all the three
                     segmentations ( sagittal_SO, coronal_SO, axial_SO ) . Stored in disk in 3d numpy *.npy format
    """
    x = np.load(sagittal_SO)
    y = np.load(coronal_SO)
    z = np.load(axial_SO)

    m, n = x.shape[::2]
    x = x.transpose(0, 3, 1, 2).reshape(m, -1, n)

    m, n = y.shape[::2]
    y = y.transpose(0, 3, 1, 2).reshape(m, -1, n)

    m, n = z.shape[::2]
    z = z.transpose(0, 3, 1, 2).reshape(m, -1, n)

    sagittal_view = list(x.ravel())
    coronal_view = list(y.ravel())
    axial_view = list(z.ravel())

    sagittal = []
    coronal = []
    axial = []

    print("Performing Muti View Aggregation...")
    for i in range(0, len(sagittal_view)):
        vector_sagittal = [1 - sagittal_view[i], sagittal_view[i]]
        vector_coronal = [1 - coronal_view[i], coronal_view[i]]
        vector_axial = [1 - axial_view[i], axial_view[i]]

        sagittal.append(np.array(vector_sagittal))
        coronal.append(np.array(vector_coronal))
        axial.append(np.array(vector_axial))

    case_name = os.path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NHDR) + 1)] + '-multi-mask.npy'
    output_file = os.path.join(os.path.dirname(input_file), output_name)

    prob_vector = []

    for i in range(0, len(sagittal_view)):
        val = np.argmax(0.4 * coronal[i] + 0.5 * axial[i] + 0.1 * sagittal[i])
        prob_vector.append(val)

    data = np.array(prob_vector)
    shape = (256, 256, 256)
    SO = data.reshape(shape)
    SO = SO.astype('float32')
    #SO = scipy.ndimage.rotate(SO, 180, axes=(0, 1))
    np.save(output_file, SO)
    return output_file


def multi_view_fast(sagittal_SO, coronal_SO, axial_SO, input_file):
    x = np.load(sagittal_SO)
    y = np.load(coronal_SO)
    z = np.load(axial_SO)

    m, n = x.shape[::2]
    x = x.transpose(0, 3, 1, 2).reshape(m, -1, n)

    m, n = y.shape[::2]
    y = y.transpose(0, 3, 1, 2).reshape(m, -1, n)

    m, n = z.shape[::2]
    z = z.transpose(0, 3, 1, 2).reshape(m, -1, n)

    x = np.multiply(x, 0.1)
    y = np.multiply(y, 0.4)
    z = np.multiply(z, 0.5)

    print("Performing Muti View Aggregation...")
    XplusY = np.add(x, y)
    multi_view = np.add(XplusY, z)
    multi_view[multi_view > 0.45] = 1
    multi_view[multi_view <= 0.45] = 0

    case_name = os.path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NHDR) + 1)] + '-multi-mask.npy'
    output_file = os.path.join(os.path.dirname(input_file), output_name)

    SO = multi_view.astype('float32')
    np.save(output_file, SO)
    return output_file


def check_gradient(Nhdr_file):
    """
    Parameters
    ----------
    Nhdr_file : str
                Accepts Input filename in Nhdr format
    Returns
    -------
    None
    """
    input_file = Nhdr_file
    header_gradient = 0
    total_gradient = 1
    bashCommand1 = "unu head " + input_file + " | grep -i sizes | awk '{print $5}'"
    bashCommand2 = "unu head " + input_file + " | grep -i _gradient_ | wc -l"
    output1 = subprocess.check_output(bashCommand1, shell=True)
    output2 = subprocess.check_output(bashCommand2, shell=True)
    if output1.strip():
        header_gradient = int(output1.decode(sys.stdout.encoding))
        total_gradient = int(output2.decode(sys.stdout.encoding))

        if header_gradient == total_gradient:
            print "Gradient check passed, ", input_file
        else:
            print "Gradient check failed, ", input_file, 'Please check file header'
            sys.exit(1)
    else:
        print "Gradient check passed, ", input_file
        return True


def resample(nii_file):
    """
    Parameters
    ----------
    nii_file    : str
                  Accepts nifti filename in *.nii.gz format
    Returns
    -------
    output_file : str
                  linear interpolated filename which is stored in disk in *.nii.gz format
    """
    print "Performing linear interpolation"

    input_file = nii_file
    case_name = os.path.basename(input_file)
    output_name = 'Comp_' + case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-linear.nii.gz'
    output_file = os.path.join(os.path.dirname(input_file), output_name)
    bashCommand_resample = "ResampleImage 3 " + input_file + " " + output_file + " " + "256x246x246 1"
    output2 = subprocess.check_output(bashCommand_resample, shell=True)
    return output_file


def get_dimension(nii_file):
    """
    Parameters
    ---------
    nii_file   : str
                 Accepts nifti filename in *.nii.gz format
    Returns
    -------
    dimensions : tuple
                 Dimension of the nifti file
                 example (128,176,256)
    """
    input_file = nii_file
    img = nib.load(input_file)
    header = img.header
    dimensions = header['dim']
    dim1 = str(dimensions[1])
    dim2 = str(dimensions[2])
    dim3 = str(dimensions[3])
    dimensions = (dim1, dim2, dim3)
    return dimensions


def extract_b0(input_file):
    """
    Parameters
    ---------
    Nhdr_file   : str
                  Accepts nhdr filename in *.nhdr format
    Returns
    --------
    output_file : str
                  Extracted b0 nhdr filename which is stored in disk
                  Uses "bse.sh" program
    """
    print "Extracting b0"
    case_dir = os.path.dirname(input_file)
    case_name = os.path.basename(input_file)
    output_name = 'dwib0_' + case_name
    output_file = os.path.join(os.path.dirname(input_file), output_name)
    if case_name.endswith(SUFFIX_NRRD) | case_name.endswith(SUFFIX_NHDR):
        bashCommand = 'bse.sh -i ' + input_file + ' -o ' + output_file + ' &>/dev/null'
    else:
        if case_name.endswith(SUFFIX_NIFTI_GZ):
            case_prefix = case_name[:len(case_name) - len(SUFFIX_NIFTI_GZ)]
        else:
            case_prefix = case_name[:len(case_name) - len(SUFFIX_NIFTI)]

        bvec_file = case_dir + '/' + case_prefix + 'bvec'
        bval_file = case_dir + '/' + case_prefix + 'bval'

        if path.exists(bvec_file):
            print "File exist ", bvec_file
        else:
            print "File not found ", bvec_file
            sys.exit(1)

        if path.exists(bval_file):
            print "File exist ", bval_file
        else:
            print "File not found ", bval_file
            sys.exit(1)

        # dwiextract only works for nifti files
        bashCommand = 'dwiextract -force -fslgrad ' + bvec_file + ' ' + bval_file + ' -bzero ' + \
                      input_file + ' ' + output_file + ' &>/dev/null'

    output = subprocess.check_output(bashCommand, shell=True)
    return output_file


def nhdr_to_nifti(Nhdr_file):
    """
    Parameters
    ---------
    Nhdr_file   : str
                  Accepts nhdr filename in *.nhdr format
    Returns
    --------
    output_file : str
                  Converted nifti file which is stored in disk
                  Uses "ConvertBetweenFilename" program
    """
    print "Converting nhdr to nifti"
    input_file = Nhdr_file
    case_name = os.path.basename(input_file)
    output_name = case_name[:len(case_name) - len(SUFFIX_NHDR)] + 'nii.gz'
    output_file = os.path.join(os.path.dirname(input_file), output_name)
    bashCommand = 'ConvertBetweenFileFormats ' + input_file + " " + output_file
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output_file


def normalize(b0_resampled):
    """
    Parameters
    ---------
    b0_resampled : str
                   Accepts b0 resampled filename in *.nii.gz format
    Returns
    --------
    output_file : str
                  Normalized by 99th percentile filename which is stored in disk
    """
    print "Normalizing input data"

    input_file = b0_resampled
    case_name = os.path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-normalized.nii.gz'
    output_file = os.path.join(os.path.dirname(input_file), output_name)
    img = nib.load(b0_resampled)
    imgU16 = img.get_data().astype(np.float32)
    p = np.percentile(imgU16, 99)
    data = imgU16 / p
    data[data > 1] = 1
    data[data < 0] = sys.float_info.epsilon
    npad = ((0, 0), (5, 5), (5, 5))
    image = np.pad(data, pad_width=npad, mode='constant', constant_values=0)
    image_dwi = nib.Nifti1Image(image, img.affine, img.header)
    nib.save(image_dwi, output_file)
    return output_file


def save_nifti(fname, data, affine=None, hdr=None):
   
    hdr.set_data_dtype('int16')
    result_img = nib.Nifti1Image(data, affine, header=hdr)
    result_img.to_filename(fname)


def binary_dilation_and_erosion(affined_mask, fname):

    print "Performing Binary dilation and erosion..."

    data_affine = nib.load(affined_mask)

    data_affine_mask = nib.load(affined_mask).get_data()

    data_affine_bool = nd.binary_dilation(data_affine_mask)

    dilated_affine = nd.binary_dilation(input=data_affine_bool, iterations=2)

    eroded_affine = nd.binary_erosion(input=dilated_affine, iterations=1)

    data_affine_binary = nd.binary_erosion(eroded_affine).astype(data_affine_mask.dtype)

    result_img = nib.Nifti1Image(data_affine_binary, data_affine.affine, data_affine.header)

    result_img.to_filename(fname)


def npy_to_nhdr(b0_normalized_cases, cases_mask_arr, sub_name, dim, view='default', reference='default', omat='default', rigid=False):
    """
    Parameters
    ---------
    b0_normalized_cases : str or list
                          str  (b0 normalized single filename which is in *.nii.gz format)
                          list (b0 normalized list of filenames which is in *.nii.gz format)
    case_mask_arr       : str or list
                          str  (single predicted mask filename which is in 3d numpy *.npy format)
                          list (list of predicted mask filenames which is in 3d numpy *.npy format)
    sub_name            : str or list
                          str  (single input case filename which is in *.nhdr format)
                          list (list of input case filename which is in *.nhdr format)
    dim                 : tuple or list of tuple
                          tuple (dimension of single case in tuple format, (128,176,256))
                          list of tuples (dimension of all cases)
    view                : str
                          Three principal axes ( Sagittal, Coronal and Axial )

    reference           : str or list
                          str  (Linear-normalized case name which is in *.nii.gz format. 
                                This is the file before the rigid-body transformation step)
    Returns
    --------
    output_mask         : str or list
                          str  (single brain mask filename which is stored in disk in *.nhdr format)
                          list (list of brain mask for all cases which is stored in disk in *.nhdr format)
    """
    print("Converting file format...")
    if isinstance(b0_normalized_cases, list):
        output_mask = []
        for i in range(0, len(b0_normalized_cases)):
            image_space = nib.load(b0_normalized_cases[i])
            predict = np.load(cases_mask_arr[i])
            predict[predict >= 0.5] = 1
            predict[predict < 0.5] = 0
            predict = predict.astype('int16')
            image_predict = nib.Nifti1Image(predict, image_space.affine, image_space.header)
            output_dir = os.path.dirname(sub_name[i])
            output_file = cases_mask_arr[i][:len(cases_mask_arr[i]) - len(SUFFIX_NPY)] + 'nii.gz'
            nib.save(image_predict, output_file)

            #print "Applying Inverse transform before downsampling"
            if rigid:
                output_file_inverseMask = inverse_transform(output_file, reference[i], omat[i])
                output_file = reThreshold(output_file_inverseMask)

            downsample_file = output_file[:len(output_file) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-downsampled.nii.gz'
            bashCommand_downsample = "ResampleImage 3 " + output_file + " " + downsample_file + " " + dim[i][0] + "x" + \
                                     dim[i][1] + "x" + dim[i][2] + " 1"
            output2 = subprocess.check_output(bashCommand_downsample, shell=True)

            case_name = os.path.basename(downsample_file)
            fill_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-filled.nii.gz'
            filled_file = os.path.join(output_dir, fill_name)
            fill_cmd = "ImageMath 3 " + filled_file + " FillHoles " + downsample_file
            process = subprocess.Popen(fill_cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

            subject_name = os.path.basename(sub_name[i])
            if subject_name.endswith(SUFFIX_NIFTI_GZ):
                format = SUFFIX_NIFTI_GZ
            else:
                format = SUFFIX_NIFTI

            # Neural Network Predicted Mask
            output_nhdr = subject_name[:len(subject_name) - (len(format) + 1)] + '-' + view + '_originalMask.nii.gz'
            output_folder = os.path.join(output_dir, output_nhdr)
            #bashCommand = 'ConvertBetweenFileFormats ' + filled_file + " " + output_folder
            bashCommand = 'mv ' + filled_file + " " + output_folder
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

            data_mask = nib.load(output_folder).get_data()
            data_dwi = nib.load(sub_name[i])

            # DWI original Image affine is copied to the Predicted Mask
            output_nhdr_affined = subject_name[:len(subject_name) - (len(format) + 1)] + '-' + view + '_affinedMask.nii.gz'
            output_folder_affined = os.path.join(output_dir, output_nhdr_affined)

            save_nifti(output_folder_affined, data_mask, affine=data_dwi.affine, hdr=data_dwi.header)

            output_nhdr_final = subject_name[:len(subject_name) - (len(format) + 1)] + '-' + view + '_BrainMask.nii.gz'
            output_folder_final = os.path.join(output_dir, output_nhdr_final)

            # After affine is copied from Original DWI Image to the Predicted Mask, Dilation is performed
            binary_dilation_and_erosion(output_folder_affined, output_folder_final)

            output_mask.append(output_folder_final)
    else:
        image_space = nib.load(b0_normalized_cases)
        predict = np.load(cases_mask_arr)
        predict[predict >= 0.5] = 1
        predict[predict < 0.5] = 0
        predict = predict.astype('int16')
        output_dir = os.path.dirname(b0_normalized_cases)
        case_mask_name = os.path.basename(cases_mask_arr)
        image_predict = nib.Nifti1Image(predict, image_space.affine, image_space.header)
        output_name = case_mask_name[:len(case_mask_name) - len(SUFFIX_NPY)] + 'nii.gz'
        output_file = os.path.join(output_dir, output_name)
        nib.save(image_predict, output_file)

        #print "Applying Inverse transform before downsampling"
        if rigid:
            output_file_inverseMask = inverse_transform(output_file, reference, omat=omat)
            output_file = reThreshold(output_file_inverseMask)

        case_name = os.path.basename(output_file)
        downsample_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-downsampled.nii.gz'
        downsample_file = os.path.join(output_dir, downsample_name)
        bashCommand_downsample = "ResampleImage 3 " + output_file + " " + downsample_file + " " + dim[0] + "x" + dim[
            1] + "x" + dim[2] + " 1"
        output2 = subprocess.check_output(bashCommand_downsample, shell=True)

        case_name = os.path.basename(downsample_file)
        fill_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-filled.nii.gz'
        filled_file = os.path.join(output_dir, fill_name)
        fill_cmd = "ImageMath 3 " + filled_file + " FillHoles " + downsample_file
        process = subprocess.Popen(fill_cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        if sub_name.endswith(SUFFIX_NIFTI_GZ):
            format = SUFFIX_NIFTI_GZ
        else:
            format = SUFFIX_NIFTI

        sub_base_name = os.path.basename(sub_name)

        # Neural Network Predicted Mask
        output_mask_name = sub_base_name[:len(sub_base_name) - (len(format) + 1)] + '-' + view + '_originalMask.nii.gz'
        output_mask = os.path.join(output_dir, output_mask_name)

        bashCommand = 'mv ' + filled_file + " " + output_mask
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        data_mask = nib.load(output_mask).get_data().astype(np.float32)
        data_dwi = nib.load(sub_name)

        # DWI original Image affine is copied to the Predicted Mask
        output_mask_name_affined = sub_base_name[:len(sub_base_name) - (len(format) + 1)] + '-' + view + '_affinedMask.nii.gz'
        output_mask_affined = os.path.join(output_dir, output_mask_name_affined)

        save_nifti(output_mask_affined, data_mask, affine=data_dwi.affine, hdr=data_dwi.header)

        output_mask_name_final = sub_base_name[:len(sub_base_name) - (len(format) + 1)] + '-' + view + '_BrainMask.nii.gz'
        output_mask_final = os.path.join(output_dir, output_mask_name_final)

        # After affine is copied from Original DWI Image to the Predicted Mask, Dilation is performed
        binary_dilation_and_erosion(output_mask_affined, output_mask_final)

    return output_mask_final


def clear(directory):
    print "Cleaning files ..."
    for filename in os.listdir(directory):
        if filename.startswith('Comp') | filename.endswith(SUFFIX_NPY) | \
                filename.endswith('_SO.nii.gz') | filename.endswith('downsampled.nii.gz') | \
                filename.endswith('-thresholded.nii.gz') | filename.endswith('-inverse.mat') | \
                filename.endswith('_affinedMask.nii.gz') | filename.endswith('_originalMask.nii.gz') | \
                filename.endswith('multi-mask.nii.gz') | filename.endswith('-mask-inverse.nii.gz'):
                os.unlink(directory + '/' + filename)


def split(cases_file, case_arr, view='default'):
    """
    Parameters
    ---------
    cases_file : str
                 Accepts a filename which is in 3d numpy array format stored in disk
    split_dim  : list
                 Contains the "x" dim for all the cases
    case_arr   : list
                 Contain filename for all the input cases
    Returns
    --------
    predict_mask : list
                   Contains the predicted mask filename of all the cases which is stored in disk in *.npy format
    """


    count = 0
    start = 0
    end = start + 256
    SO = np.load(cases_file)

    predict_mask = []
    for i in range(0, len(case_arr)):
        end = start + 256
        casex = SO[start:end, :, :]
        if view == 'coronal':
            casex = np.swapaxes(casex, 0, 1)
        elif view == 'axial':
            casex = np.swapaxes(casex, 0, 2)
        input_file = str(case_arr[i])
        output_file = input_file[:len(input_file) - (len(SUFFIX_NHDR) + 1)] + '-' + view +'_SO.npy'
        predict_mask.append(output_file)
        np.save(output_file, casex)
        start = end
        count += 1

    return predict_mask


def rigid_body_trans(b0_nii):

    print("Performing rigid body transformation...")
    input_file = b0_nii
    case_name = os.path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-transformed.nii.gz'
    output_file = os.path.join(os.path.dirname(input_file), output_name)

    reference = '/rfanfs/pnl-zorro/home/sq566/CompNetPipeline/reference/eight256.nii.gz'

    #Compute Transformation matrix using flirt
    omat_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '.mat'
    omat_file = os.path.join(os.path.dirname(input_file), omat_name)

    trans_matrix = "flirt -in " + input_file +  " -ref " + reference + \
                   " -omat " + omat_file + " -dof 6 -cost mutualinfo"

    #print trans_matrix
    output1 = subprocess.check_output(trans_matrix, shell=True)

    # Apply this transformation to the input volume
    apply_trans = "flirt -in " + input_file + " -ref " + reference + \
                  " -applyxfm -init " + omat_file + " -o " + output_file

    #print apply_trans
    output2 = subprocess.check_output(apply_trans, shell=True)

    return output_file, omat_file


def inverse_transform(predicted_mask, reference, omat='default'):
    """
    Parameters
    ---------
    predicted_mask :      str
                          str  (Neural Network mask filename which is in *.nii.gz format)
    reference      :      str
                          str  (Linear-normalized case name which is in *.nii.gz format . 
                                This is the file before the rigid-body transformation step)
    Returns
    --------
    output_mask    :      str
                          str  (Inverse transformed brain mask filename which is stored in disk in *.nii.gz format)
    """

    print("Performing inverse transform...")
    input_file = predicted_mask
    case_name = os.path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-inverse.nii.gz'
    output_file = os.path.join(os.path.dirname(input_file), output_name)

    # Invert the matrix
    omat_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-inverse.mat'
    omat_file = os.path.join(os.path.dirname(input_file), omat_name)

    inverse = "convert_xfm -omat " + omat_file + " -inverse " + omat
    output1 = subprocess.check_output(inverse, shell=True)
    
    # Apply the inverse transformation to the predicted mask
    # print "omat_rigid = ", omat
    # print "Reference =", reference
    # print "omat_inverse =", omat_file

    apply_inverse_trans = "flirt -in " + input_file + " -ref " + reference + " -applyxfm -init " + omat_file + " -o " + output_file
    output2 = subprocess.check_output(apply_inverse_trans, shell=True)

    return output_file


def reThreshold(binary_mask):
    """
    Only voxels in the new space that overlap by 90% with the 
    original mask will be included in the new binary mask
    """
    print("Thresholding binary mask...")
    input_file = binary_mask
    case_name = os.path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-thresholded.nii.gz'
    output_file = os.path.join(os.path.dirname(input_file), output_name)

    threshold = "fslmaths " + input_file +  " -thr 0.9 -bin " + output_file
    output1 = subprocess.check_output(threshold, shell=True)

    return output_file


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def list_masks(mask_list, view='default'):

    for i in range(0, len(mask_list)):
        print view + " Mask file = ", mask_list[i]


if __name__ == '__main__':

    start_t = datetime.datetime.now()
    # parser module for input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', action='store', dest='dwi', type=str,
                        help=" input single diffusion image or a caselist ")

    parser.add_argument("-axial", type=str2bool, dest='Axial', nargs='?',
                        const=True, default=False,
                        help="Activate axial Mask (yes/true/y/1)")

    parser.add_argument("-coronal", type=str2bool, dest='Coronal', nargs='?',
                        const=True, default=False,
                        help="Activate coronal Mask (yes/true/y/1)")

    parser.add_argument("-sagittal", type=str2bool, dest='Sagittal', nargs='?',
                        const=True, default=False,
                        help="Activate sagittal Mask (yes/true/y/1)")

    parser.add_argument("-rigid", type=str2bool, dest='Rigid', nargs='?',
                        const=True, default=False,
                        help="Pefrorm Rigid Body Transformation (yes/true/y/1)")

    try:
        args = parser.parse_args()
        if len(sys.argv) == 1:
            parser.print_help()
            parser.error('too few arguments')
            sys.exit(0)

    except SystemExit:
        sys.exit(0)



    tmp_path = "/rfanfs/pnl-zorro/home/sq566/tmp"

    if args.dwi:
        f = pathlib.Path(args.dwi)
        if f.exists():
            print "File exist"
            filename = args.dwi
        else:
            print "File not found"
            sys.exit(1)

        # Input caselist.txt
        if filename.endswith(SUFFIX_TXT):
            with open(filename) as f:
                case_arr = f.read().splitlines()


            TXT_file = os.path.basename(filename)
            #print(TXT_file)
            unique = TXT_file[:len(TXT_file) - (len(SUFFIX_TXT)+1)]
            #print(unique)
            binary_file_s = '/rfanfs/pnl-zorro/home/sq566/tmp/' + unique + '_binary_s'
            binary_file_c = '/rfanfs/pnl-zorro/home/sq566/tmp/'+ unique + '_binary_c'
            binary_file_a = '/rfanfs/pnl-zorro/home/sq566/tmp/'+ unique + '_binary_a'

            f_handle_s = open(binary_file_s, 'wb')
            f_handle_c = open(binary_file_c, 'wb')
            f_handle_a = open(binary_file_a, 'wb')

            x_dim = 0
            y_dim = 256
            z_dim = 256
            split_dim = []
            b0_normalized_cases = []
            cases_dim = []
            reference_list = []
            omat_list = []
            count = 0

            for subjects in case_arr:
                input_file = subjects
                f = pathlib.Path(input_file)

                if f.exists():
                    input_file = str(f)
                    asb_path = os.path.abspath(input_file)
                    directory = os.path.dirname(input_file)
                    input_file = os.path.basename(input_file)

                    if input_file.endswith(SUFFIX_NRRD) | input_file.endswith(SUFFIX_NHDR):
                        if not check_gradient(os.path.join(directory, input_file)):
                           b0_nhdr = extract_b0(os.path.join(directory, input_file))
                        else:
                           b0_nhdr = os.path.join(directory, input_file)

                        b0_nii = nhdr_to_nifti(b0_nhdr)
                    else:
                        b0_nii = extract_b0(os.path.join(directory, input_file))

                    dimensions = get_dimension(b0_nii)
                    cases_dim.append(dimensions)
                    x_dim += int(dimensions[0])
                    split_dim.append(int(dimensions[0]))
                    b0_resampled = resample(b0_nii)
                    b0_normalized = normalize(b0_resampled)
                    reference_list.append(b0_normalized)

                    if args.Rigid:
                        b0_transform, omat_file = rigid_body_trans(b0_normalized)
                        b0_normalized_cases.append(b0_transform)
                        omat_list.append(omat_file)
                        img = nib.load(b0_transform)
                    else:
                        b0_normalized_cases.append(b0_normalized)
                        img = nib.load(b0_normalized)

                    imgU16_sagittal = img.get_data().astype(np.float32)  # sagittal view

                    imgU16_coronal = np.swapaxes(imgU16_sagittal, 0, 1)  # coronal view

                    imgU16_axial = np.swapaxes(imgU16_sagittal, 0, 2) # Axial view

                    imgU16_sagittal.tofile(f_handle_s)
                    imgU16_coronal.tofile(f_handle_c)
                    imgU16_axial.tofile(f_handle_a)
                    print "Case completed = ", count
                    count += 1

                else:
                    print "File not found ", input_file
                    sys.exit(1)
            f_handle_s.close()
            f_handle_c.close()
            f_handle_a.close()
            print "Merging npy files"
            cases_file_s = '/rfanfs/pnl-zorro/home/sq566/tmp/'+ unique + '-casefile-sagittal.npy'
            cases_file_c = '/rfanfs/pnl-zorro/home/sq566/tmp/'+ unique + '-casefile-coronal.npy'
            cases_file_a = '/rfanfs/pnl-zorro/home/sq566/tmp/'+ unique + '-casefile-axial.npy'

            merge_s = np.memmap(binary_file_s, dtype=np.float32, mode='r+', shape=(256 * len(cases_dim), y_dim, z_dim))
            merge_c = np.memmap(binary_file_c, dtype=np.float32, mode='r+', shape=(256 * len(cases_dim), y_dim, z_dim))
            merge_a = np.memmap(binary_file_a, dtype=np.float32, mode='r+', shape=(256 * len(cases_dim), y_dim, z_dim))

            print "Saving training data to disk"
            np.save(cases_file_s, merge_s)
            np.save(cases_file_c, merge_c)
            np.save(cases_file_a, merge_a)

            dwi_mask_sagittal = predict_mask(cases_file_s, view='sagittal')
            dwi_mask_coronal = predict_mask(cases_file_c, view='coronal')
            dwi_mask_axial = predict_mask(cases_file_a, view='axial')

            print "Splitting files...."

            cases_mask_sagittal = split(dwi_mask_sagittal, case_arr, view='sagittal')
            cases_mask_coronal = split(dwi_mask_coronal, case_arr, view='coronal')
            cases_mask_axial = split(dwi_mask_axial, case_arr, view='axial')

            slices = " "
            for i in range(0, len(cases_mask_sagittal)):

                sagittal_SO = cases_mask_sagittal[i]
                coronal_SO = cases_mask_coronal[i]
                axial_SO = cases_mask_axial[i]

                input_file = case_arr[i]

                #multi_view_mask = multi_view_agg(sagittal_SO, coronal_SO, axial_SO, input_file)
                multi_view_mask = multi_view_fast(sagittal_SO, 
                                                  coronal_SO, 
                                                  axial_SO, 
                                                  input_file)

                if args.Rigid:
                    brain_mask_multi = npy_to_nhdr(b0_normalized_cases[i], 
                                                    multi_view_mask, 
                                                    case_arr[i], 
                                                    cases_dim[i],
                                                    view='multi', 
                                                    reference=reference_list[i], 
                                                    omat=omat_list[i], 
                                                    rigid=args.Rigid)
                else:
                      brain_mask_multi = npy_to_nhdr(b0_normalized_cases[i], 
                                                    multi_view_mask, 
                                                    case_arr[i], 
                                                    cases_dim[i],
                                                    view='multi', 
                                                    rigid=args.Rigid)

                print "Mask file = ", brain_mask_multi

                str1 = case_arr[i]
                str2 = brain_mask_multi
                slices += str1 + " " + str2 + " "

            final = "slicesdir -o" + slices
            #print final
        
            os.chdir(tmp_path)
            subprocess.check_output(final, shell=True)

            if args.Sagittal:
                sagittal_mask = npy_to_nhdr(b0_normalized_cases, 
                                            cases_mask_sagittal, 
                                            case_arr, 
                                            cases_dim, 
                                            view='sagittal', 
                                            omat=omat_list, 
                                            rigid=args.Rigid)
                list_masks(sagittal_mask, view='sagittal')

            if args.Coronal:
                coronal_mask = npy_to_nhdr(b0_normalized_cases, 
                                           cases_mask_coronal, 
                                           case_arr, 
                                           cases_dim, 
                                           view='coronal', 
                                           omat=omat_list, 
                                           rigid=args.Rigid)
                list_masks(coronal_mask, view='coronal')

            if args.Axial:
                axial_mask = npy_to_nhdr(b0_normalized_cases, 
                                         cases_mask_axial, 
                                         case_arr, 
                                         cases_dim, 
                                         view='axial', 
                                         omat=omat_list, 
                                         rigid=args.Rigid)
                list_masks(axial_mask, view='axial')

            clear(os.path.dirname(brain_mask_multi))
            webbrowser.open(os.path.join(tmp_path, 'slicesdir/index.html'))

        # Input in nrrd / nhdr / nii / nii.gz format
        elif filename.endswith(SUFFIX_NHDR) | filename.endswith(SUFFIX_NRRD) | filename.endswith(SUFFIX_NIFTI_GZ):
          
            input_file = filename
            asb_path = os.path.abspath(input_file)
            directory = os.path.dirname(asb_path)
            input_file = os.path.basename(asb_path)

            if input_file.endswith(SUFFIX_NRRD) | input_file.endswith(SUFFIX_NHDR):
                if not check_gradient(os.path.join(directory, input_file)):
                    b0_nhdr = extract_b0(os.path.join(directory, input_file))
                else:
                    b0_nhdr = os.path.join(directory, input_file)

                b0_nii = nhdr_to_nifti(b0_nhdr)
            else:
                b0_nii = extract_b0(os.path.join(directory, input_file))

            dimensions = get_dimension(b0_nii)
            b0_resampled = resample(b0_nii)
            b0_normalized = normalize(b0_resampled)

            if args.Rigid:
                b0_transform, omat_file = rigid_body_trans(b0_normalized)
            else:
                b0_transform = b0_normalized
                omat_file = None

            dwi_mask_sagittal = predict_mask(b0_transform, view='sagittal')
            dwi_mask_coronal = predict_mask(b0_transform, view='coronal')
            dwi_mask_axial = predict_mask(b0_transform, view='axial')

            subject_name = os.path.join(directory, input_file)

            multi_view_mask = multi_view_fast(dwi_mask_sagittal, 
                                             dwi_mask_coronal, 
                                             dwi_mask_axial, 
                                             input_file)

            if args.Sagittal:
                brain_mask_sagittal = npy_to_nhdr(b0_transform, 
                                                  dwi_mask_sagittal, 
                                                  subject_name,
                                                  dimensions, 
                                                  view='sagittal', 
                                                  reference=b0_normalized,
                                                  omat=omat_file, rigid=args.Rigid)
            if args.Coronal:
                brain_mask_coronal = npy_to_nhdr(b0_transform, 
                                                dwi_mask_coronal, 
                                                subject_name, 
                                                dimensions,
                                                view='coronal', 
                                                reference=b0_normalized,
                                                omat=omat_file, rigid=args.Rigid)
            if args.Axial:
                brain_mask_axial = npy_to_nhdr(b0_transform, 
                                               dwi_mask_axial, 
                                               subject_name, 
                                               dimensions, 
                                               view='axial', 
                                               reference=b0_normalized,
                                               omat=omat_file, rigid=args.Rigid)

            brain_mask_multi = npy_to_nhdr(b0_transform, 
                                            multi_view_mask, 
                                            subject_name, 
                                            dimensions, 
                                            view='multi', 
                                            reference=b0_normalized, 
                                            omat=omat_file, rigid=args.Rigid)
            clear(directory)
            print "Multiview Mask file = ", brain_mask_multi
            if args.Sagittal:
                print "Sagittal Mask file = ", brain_mask_sagittal
            if args.Coronal:
                print "Coronal Mask file = ", brain_mask_coronal
            if args.Axial:
                print "Axial Mask file = ", brain_mask_axial

            slices = " "
            str1 = subject_name
            str2 = brain_mask_multi
            slices += str1 + " " + str2 + " "

            final = "slicesdir -o" + slices
            #print final
            os.chdir(tmp_path)
            subprocess.check_output(final, shell=True)
            webbrowser.open(os.path.join(tmp_path, 'slicesdir/index.html'))

        end_t = datetime.datetime.now()
        total_t = end_t - start_t
print("Time Taken in sec = ", total_t.seconds)
