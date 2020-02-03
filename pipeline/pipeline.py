from __future__ import division
# -----------------------------------------------------------------
# Author:       PNL BWH                 
# Written:      07/02/2019                             
# Last Updated:     01/15/2020
# Purpose:          Python pipeline for diffusion brain masking
# -----------------------------------------------------------------

"""
pipeline.py
~~~~~~~~~~
01)  Accepts the diffusion image in *.nhdr,*.nrrd,*.nii.gz,*.nii format
02)  Checks if the Image axis is in the correct order for *.nhdr and *.nrrd file
03)  Extracts b0 Image
04)  Converts nhdr to nii.gz
05)  Applys Rigid-Body tranformation to standard MNI space using
06)  Normalize the Image by 99th percentile
07)  Neural network brain mask prediction across the 3 principal axis
08)  Performs Multi View Aggregation
09)  Converts npy to nhdr,nrrd,nii,nii.gz
10)  Applys Inverse tranformation
11)  Cleaning
"""


# pylint: disable=invalid-name
import os
import os.path
from os import path
import webbrowser
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress tensor flow message
import GPUtil 

# Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Get the first available GPU
try:
    DEVICE_ID_LIST = GPUtil.getFirstAvailable()
    DEVICE_ID = DEVICE_ID_LIST[0] # Grab first element from list
    print ("GPU found...", DEVICE_ID)

    # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

except RuntimeError:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("GPU not available...")

import tensorflow as tf
import multiprocessing as mp
import re
import sys
import subprocess
import argparse, textwrap
import datetime
import os.path
import pathlib
import nibabel as nib
import numpy as np
import scipy.ndimage as nd
from os import path
from keras.models import load_model
from keras.models import model_from_json
from keras.utils import multi_gpu_model
from multiprocessing import Process, Manager, Value, Pool
import multiprocessing as mp
import sys
from time import sleep
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
output_mask = []


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
    print ("Loading " + view + " model from disk...")
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
    json_file = open('/rfanfs/pnl-zorro/software/CNN-Diffusion-BrainMask-Trained-Model-Suheyla/CompNetBasicModel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    optimal = ''
    if view == 'sagittal':
        optimal = '09'
    elif view == 'coronal':
        optimal = '08'
    else:
        optimal = '08'
    # load weights into new model
    loaded_model.load_weights(
        '/rfanfs/pnl-zorro/software/CNN-Diffusion-BrainMask-Trained-Model-Suheyla/weights-' + view + '-improvement-' + optimal + '.h5')

    # evaluate loaded model on test data
    loaded_model.compile(optimizer=Adam(lr=1e-5),
                         loss={'final_op': dice_coef_loss,
                               'xfinal_op': neg_dice_coef_loss,
                               'res_1_final_op': 'mse'})

    case_name = os.path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-' + view + '-mask.npy'
    output_file = os.path.join(os.path.dirname(input_file), output_name)

    x_test = np.load(input_file)
    x_test = x_test.reshape(x_test.shape + (1,))
    predict_x = loaded_model.predict(x_test, verbose=1)
    SO = predict_x[0]  # Segmentation Output
    del predict_x
    np.save(output_file, SO)
    return output_file


def multi_view_slow(sagittal_SO, coronal_SO, axial_SO, input_file):
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
            print ("Gradient check passed, ", input_file)
        else:
            print ("Gradient check failed, ", input_file, 'Please check file header')
            sys.exit(1)
    else:
        print ("Gradient check passed, ", input_file)
        return True


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
    input_file   : str
                  Accepts nhdr filename in *.nhdr format
                  Accepts nifti filename in *.nii.gz format
    Returns
    --------
    output_file : str
                  Extracted b0 nhdr filename which is stored in disk
                  Uses "bse.sh" program
    """
    print ("Extracting b0...")
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
            print ("File exist ", bvec_file)
        else:
            print ("File not found ", bvec_file)
            bvec_file = case_dir + '/' + case_prefix + 'bvecs'
            if path.exists(bvec_file):
                print ("File exist ", bvec_file)
            else:
                print ("File not found ", bvec_file)
            sys.exit(1)

        if path.exists(bval_file):
            print ("File exist ", bval_file)
        else:
            print ("File not found ", bval_file)
            bval_file = case_dir + '/' + case_prefix + 'bvals'
            if path.exists(bval_file):
                print ("File exist ", bval_file)
            else:
                print ("File not found ", bval_file)
            sys.exit(1)

        # dwiextract only works for nifti files
        dwiextract = 'dwiextract -force -fslgrad ' + bvec_file + ' ' + bval_file + ' -bzero ' + \
                      input_file + ' ' + output_file + ' &>/dev/null'
        #dwiextract = 'dwiextract -force -fslgrad ' + bvec_file + ' ' + bval_file + ' -shell 900 ' + \
        #              input_file + ' ' + output_file + ' &>/dev/null'

        subprocess.check_output(dwiextract, shell=True)

        bashCommand = 'mrmath -force ' + output_file + " mean " + output_file + " -axis 3" + ' &>/dev/null'

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
    print ("Converting nhdr to nifti")
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
    Intensity based segmentation of MR images is hampered by radio frerquency field
    inhomogeneity causing intensity variation. The intensity range is typically
    scaled between the highest and lowest signal in the Image. Intensity values
    of the same tissue can vary between scans. The pixel value in images must be
    scaled prior to providing the images as input to CNN. The data is projected in to
    a predefined range [0,1]

    Parameters
    ---------
    b0_resampled : str
                   Accepts b0 resampled filename in *.nii.gz format
    Returns
    --------
    output_file : str
                  Normalized by 99th percentile filename which is stored in disk
    """
    print ("Normalizing input data")

    input_file = b0_resampled
    case_name = os.path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-normalized.nii.gz'
    output_file = os.path.join(os.path.dirname(input_file), output_name)
    img = nib.load(b0_resampled)
    imgU16 = img.get_data().astype(np.float32)
    p = np.percentile(imgU16, 99)
    data = imgU16 / p
    data[data > 1] = 1
    data[data < 0] = 0
    image_dwi = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(image_dwi, output_file)
    return output_file


def save_nifti(fname, data, affine=None, hdr=None):
   
    hdr.set_data_dtype('int16')
    result_img = nib.Nifti1Image(data, affine, header=hdr)
    result_img.to_filename(fname)


def npy_to_nhdr(b0_normalized_cases, cases_mask_arr, sub_name, view='default', reference='default', omat=None):
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
    global output_mask
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

        output_file_inverseMask = ANTS_inverse_transform(output_file, reference[i], omat[i])
        output_file = output_file_inverseMask

        case_name = os.path.basename(output_file)
        fill_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-filled.nii.gz'
        filled_file = os.path.join(output_dir, fill_name)
        fill_cmd = "ImageMath 3 " + filled_file + " FillHoles " + output_file
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
        bashCommand = 'mv ' + filled_file + " " + output_folder
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        output_filter_folder = subject_name[:len(subject_name) - (len(format) + 1)] + '-' + view + '_FilteredMask.nii.gz'
        output_mask_filtered = os.path.join(output_dir, output_filter_folder)

        mask_filter = "maskfilter -force " + output_folder + " -scale 2 clean " + output_mask_filtered
        process = subprocess.Popen(mask_filter.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        img = nib.load(output_mask_filtered)
        data_dwi = nib.load(sub_name[i])
        imgU16 = img.get_data().astype(np.int16)

        output_folder_final = subject_name[:len(subject_name) - (len(format) + 1)] + '-' + view + '_BrainMask.nii.gz'
        output_mask_final = os.path.join(output_dir, output_folder_final)

        save_nifti(output_mask_final, imgU16, affine=data_dwi.affine, hdr=data_dwi.header)
        output_mask.append(output_mask_final)

    return output_mask


def clear(directory):
    print ("Cleaning files ...")
    for filename in os.listdir(directory):
        if filename.startswith('Comp') | filename.endswith(SUFFIX_NPY) | \
                filename.endswith('_SO.nii.gz') | filename.endswith('downsampled.nii.gz') | \
                filename.endswith('-thresholded.nii.gz') | filename.endswith('-inverse.mat') | \
                filename.endswith('-Warped.nii.gz') | filename.endswith('-0GenericAffine.mat') | \
                filename.endswith('_affinedMask.nii.gz') | filename.endswith('_originalMask.nii.gz') | \
                filename.endswith('multi-mask.nii.gz') | filename.endswith('-mask-inverse.nii.gz') | \
                filename.endswith('-InverseWarped.nii.gz') | filename.endswith('-FilteredMask.nii.gz') | \
                filename.endswith('_FilteredMask.nii.gz') | filename.endswith('-normalized.nii.gz'):
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


def ANTS_rigid_body_trans(b0_nii, reference=None):

    print("Performing ants rigid body transformation...")
    input_file = b0_nii
    case_name = os.path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-'
    output_file = os.path.join(os.path.dirname(input_file), output_name)

    if reference is None:
        reference = '/rfanfs/pnl-zorro/software/CNN-Diffusion-BrainMask-Trained-Model-Suheyla/IITmean_b0_256.nii.gz'

    trans_matrix = "antsRegistrationSyNQuick.sh -d 3 -f " + reference + " -m " + input_file + " -t r -o " + output_file
    output1 = subprocess.check_output(trans_matrix, shell=True)

    omat_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-0GenericAffine.mat'
    omat_file = os.path.join(os.path.dirname(input_file), omat_name)

    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-Warped.nii.gz'
    transformed_file = os.path.join(os.path.dirname(input_file), output_name)

    #print "output_file = ", transformed_file
    #print "omat_file = ", omat_file
    return (transformed_file, omat_file)


def ANTS_inverse_transform(predicted_mask, reference, omat='default'):

    #print "Mask file = ", predicted_mask
    #print "Reference = ", reference
    #print "omat = ", omat

    print("Performing ants inverse transform...")
    input_file = predicted_mask
    case_name = os.path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-inverse.nii.gz'
    output_file = os.path.join(os.path.dirname(input_file), output_name)

    # reference is the original b0 volume
    apply_inverse_trans = "antsApplyTransforms -d 3 -i " + predicted_mask + " -r " + reference + " -o " \
                            + output_file + " --transform [" + omat + ",1]"

    output2 = subprocess.check_output(apply_inverse_trans, shell=True)
    return output_file


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected...')


def list_masks(mask_list, view='default'):

    for i in range(0, len(mask_list)):
        print (view + " Mask file = ", mask_list[i])


def pre_process(lock, subject, reference_list):

    #lock.acquire()
    #try:
        input_file = subject
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
                #b0_nii = os.path.join(directory, input_file)
                b0_nii = extract_b0(os.path.join(directory, input_file))

            #dimensions = get_dimension(b0_nii)
            #cases_dim.append(dimensions)
            #split_dim.append(int(dimensions[0]))
            reference_list.append((b0_nii))

        else:
            print ("File not found ", input_file)
            sys.exit(1)

    #finally:
        #lock.release()


def quality_control(mask_list, shuffled_list, tmp_path, view='default'):

    slices = " "
    for i in range(0, len(mask_list)):
        str1 = shuffled_list[i]
        str2 = mask_list[i]
        slices += str1 + " " + str2 + " "
    
    final = "slicesdir -o" + slices
    os.chdir(tmp_path)
    subprocess.check_output(final, shell=True)
    mask_folder = os.path.join(tmp_path, 'slicesdir')
    mask_newfolder = os.path.join(tmp_path, 'slicesdir_' + view)
    bashCommand = 'mv --force ' + mask_folder + " " + mask_newfolder
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()


if __name__ == '__main__':

    start_total_time = datetime.datetime.now()
    # parser module for input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', action='store', dest='dwi', type=str,
                        help=" input caselist file in txt format")

    parser.add_argument("-axial", type=str2bool, dest='Axial', nargs='?',
                        const=True, default=False,
                        help="generate axial Mask (yes/true/y/1)")

    parser.add_argument("-coronal", type=str2bool, dest='Coronal', nargs='?',
                        const=True, default=False,
                        help="generate coronal Mask (yes/true/y/1)")

    parser.add_argument("-sagittal", type=str2bool, dest='Sagittal', nargs='?',
                        const=True, default=False,
                        help="generate sagittal Mask (yes/true/y/1)")

    parser.add_argument('-nproc', type=int, dest='cr', default=8, help='number of processes to use')

    try:
        args = parser.parse_args()
        if len(sys.argv) == 1:
            parser.print_help()
            parser.error('too few arguments')
            sys.exit(0)

    except SystemExit:
        sys.exit(0)

    if args.dwi:
        f = pathlib.Path(args.dwi)
        if f.exists():
            print ("File exist")
            filename = args.dwi
        else:
            print ("File not found")
            sys.exit(1)

        # Input caselist.txt
        if filename.endswith(SUFFIX_TXT):
            with open(filename) as f:
                case_arr = f.read().splitlines()


            TXT_file = os.path.basename(filename)
            #print(TXT_file)
            unique = TXT_file[:len(TXT_file) - (len(SUFFIX_TXT)+1)]
            #print(unique)
            storage = os.path.dirname(case_arr[0])
            tmp_path = storage + '/'

            binary_file_s = storage + '/' + unique + '_binary_s'
            binary_file_c = storage + '/'+ unique + '_binary_c'
            binary_file_a = storage + '/'+ unique + '_binary_a'

            f_handle_s = open(binary_file_s, 'wb')
            f_handle_c = open(binary_file_c, 'wb')
            f_handle_a = open(binary_file_a, 'wb')

            x_dim = 0
            y_dim = 256
            z_dim = 256
            transformed_cases = []
            
            with Manager() as manager:
                # Share a list between processes using manager
                #split_dim = manager.list()
                #cases_dim = manager.list()
                reference_list = manager.list()
                omat_list = []                
                jobs = []

                lock = mp.Lock()
                for i in range(0,len(case_arr)):
                    p = mp.Process(target=pre_process, args=(lock,case_arr[i],
                                                             #split_dim, 
                                                             #cases_dim, 
                                                             reference_list))
                    p.start()
                    jobs.append(p)
        
                for process in jobs:
                    process.join()
                #print(list(reference_list))

                reference_list = list(reference_list)
                #split_dim = list(split_dim)
                #cases_dim = list(cases_dim)

            """
            Enable Multi core Processing for ANTS Registration
            """
            p = Pool(processes=args.cr)
            data = p.map(ANTS_rigid_body_trans, reference_list)
            p.close()

            for subject_ANTS in data:
                transformed_cases.append(subject_ANTS[0])
                omat_list.append(subject_ANTS[1])

            #print(transformed_cases)

            p1 = Pool(processes=mp.cpu_count())
            data_n = p1.map(normalize, transformed_cases)
            p1.close()

            #print(data_n)
            
            count = 0
            for b0_nifti in data_n:
                
                #img_normalize = normalize(b0_nifti)
                img = nib.load(b0_nifti)

                imgU16_sagittal = img.get_data().astype(np.float32)  # sagittal view
                imgU16_coronal = np.swapaxes(imgU16_sagittal, 0, 1)  # coronal view
                imgU16_axial = np.swapaxes(imgU16_sagittal, 0, 2)    # Axial view

                imgU16_sagittal.tofile(f_handle_s)
                imgU16_coronal.tofile(f_handle_c)
                imgU16_axial.tofile(f_handle_a)

                print ("Case completed = ", count)
                count += 1

            shuffled_list = []
            reference_new_list = []
            for i in range(0, len(reference_list)):
                reference_new_list.append(reference_list[i])
                shuffled_list.append(reference_list[i])
            reference_list = reference_new_list

            f_handle_s.close()
            f_handle_c.close()
            f_handle_a.close()

            print ("Merging npy files...")
            cases_file_s = storage + '/'+ unique + '-casefile-sagittal.npy'
            cases_file_c = storage + '/'+ unique + '-casefile-coronal.npy'
            cases_file_a = storage + '/'+ unique + '-casefile-axial.npy'

            merge_s = np.memmap(binary_file_s, dtype=np.float32, mode='r+', shape=(256 * len(reference_list), y_dim, z_dim))
            merge_c = np.memmap(binary_file_c, dtype=np.float32, mode='r+', shape=(256 * len(reference_list), y_dim, z_dim))
            merge_a = np.memmap(binary_file_a, dtype=np.float32, mode='r+', shape=(256 * len(reference_list), y_dim, z_dim))

            print ("Saving data to disk...")
            np.save(cases_file_s, merge_s)
            np.save(cases_file_c, merge_c)
            np.save(cases_file_a, merge_a)

            end_preprocessing_time = datetime.datetime.now()
            total_preprocessing_time = end_preprocessing_time - start_total_time
            print ("Pre-Processing Time Taken : ", round(int(total_preprocessing_time.seconds)/60, 2), " min")

            dwi_mask_sagittal = predict_mask(cases_file_s, view='sagittal')
            dwi_mask_coronal = predict_mask(cases_file_c, view='coronal')
            dwi_mask_axial = predict_mask(cases_file_a, view='axial')

            end_masking_time = datetime.datetime.now()
            total_masking_time = end_masking_time - start_total_time - total_preprocessing_time
            print ("Masking Time Taken : ", round(int(total_masking_time.seconds)/60, 2), " min")

            print ("Splitting files....")

            cases_mask_sagittal = split(dwi_mask_sagittal, shuffled_list, view='sagittal')
            cases_mask_coronal = split(dwi_mask_coronal, shuffled_list, view='coronal')
            cases_mask_axial = split(dwi_mask_axial, shuffled_list, view='axial')

            multi_mask = []
            for i in range(0, len(cases_mask_sagittal)):

                sagittal_SO = cases_mask_sagittal[i]
                coronal_SO = cases_mask_coronal[i]
                axial_SO = cases_mask_axial[i]

                input_file = shuffled_list[i]

                #multi_view_mask = multi_view_agg(sagittal_SO, coronal_SO, axial_SO, input_file)
                multi_view_mask = multi_view_fast(sagittal_SO, 
                                                  coronal_SO, 
                                                  axial_SO, 
                                                  input_file)


                brain_mask_multi = npy_to_nhdr(list(transformed_cases[i].split()), 
                                                list(multi_view_mask.split()), 
                                                list(shuffled_list[i].split()),
                                                view='multi', 
                                                reference=list(reference_list[i].split()), 
                                                omat=list(omat_list[i].split()))


                print ("Mask file : ", brain_mask_multi)
                multi_mask.append(brain_mask_multi[0])

            quality_control(multi_mask, shuffled_list, tmp_path, view='multi')

            if args.Sagittal:
                omat = omat_list
            else:
                omat = None

            if args.Sagittal:
               
                #print("one = ", transformed_cases)
                #print("two = ", cases_mask_sagittal)
                #print("three = ", shuffled_list)
                #print("four = ", reference_list)
                #print("five = ", omat)
                sagittal_mask = npy_to_nhdr(transformed_cases, 
                                            cases_mask_sagittal, 
                                            shuffled_list,
                                            view='sagittal', 
                                            reference=reference_list,
                                            omat=omat)
                list_masks(sagittal_mask, view='sagittal')
                quality_control(sagittal_mask, shuffled_list, tmp_path, view='sagittal')

            if args.Coronal:
                omat = omat_list
            else:
                omat = None

            if args.Coronal:
           
                coronal_mask = npy_to_nhdr(transformed_cases, 
                                          cases_mask_coronal, 
                                          shuffled_list,
                                          view='coronal', 
                                          reference=reference_list,
                                          omat=omat)
                list_masks(coronal_mask, view='coronal')
                quality_control(coronal_mask, shuffled_list, tmp_path, view='coronal')

            if args.Axial:
                omat = omat_list
            else:
                omat = None

            if args.Axial:
           
                axial_mask = npy_to_nhdr(transformed_cases, 
                                         cases_mask_axial, 
                                         shuffled_list,
                                         view='axial', 
                                         reference=reference_list,
                                         omat=omat)
                list_masks(axial_mask, view='axial')
                quality_control(axial_mask, shuffled_list, tmp_path, view='axial')

            for i in range(0, len(cases_mask_sagittal)):
                clear(os.path.dirname(cases_mask_sagittal[i]))

            webbrowser.open(os.path.join(tmp_path, 'slicesdir_multi/index.html'))
            if args.Sagittal:
                webbrowser.open(os.path.join(tmp_path, 'slicesdir_sagittal/index.html'))
            if args.Coronal:
                webbrowser.open(os.path.join(tmp_path, 'slicesdir_coronal/index.html'))
            if args.Axial:
                webbrowser.open(os.path.join(tmp_path, 'slicesdir_axial/index.html'))

        end_total_time = datetime.datetime.now()
        total_t = end_total_time - start_total_time
print ("Total Time Taken : ", round(int(total_t.seconds)/60, 2), " min")
