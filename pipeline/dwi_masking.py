from __future__ import division
# -----------------------------------------------------------------
# Author:       Senthil Palanivelu, Tashrif Billah                 
# Written:      01/22/2020                             
# Last Updated:     02/05/2020
# Purpose:          CNN diffusion brain masking
# -----------------------------------------------------------------

"""
pipeline.py
~~~~~~~~~~
01)  Neural network brain mask prediction across the 3 principal axis
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


def predict_mask(input_file, view='default', trained_folder):
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
    json_file = open(trained_folder + '/CompNetBasicModel.json', 'r')
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
    loaded_model.load_weights(trained_folder + '/weights-' + view + '-improvement-' + optimal + '.h5')

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


if __name__ == '__main__':

    start_total_time = datetime.datetime.now()
    # parser module for input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store', dest='dwi', type=str,
                        help=" input caselist file in txt format")
    parser.add_argument('-f', action='store', dest='model_folder', type=str,
                        help=" folder which contain the trained model")
    args = parser.parse_args()
    mask_list = []
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

    storage = os.path.dirname(case_arr[0])
    merged_file = storage + '/' + "merged_cases.txt"
    with open(merged_file) as f:
        merged_cases_npy = f.read().splitlines()

    dwi_mask_sagittal = predict_mask(merged_cases_npy[0], view='sagittal', args.model_folder)
    dwi_mask_coronal = predict_mask(merged_cases_npy[1], view='coronal', args.model_folder)
    dwi_mask_axial = predict_mask(merged_cases_npy[2], view='axial', args.model_folder)

    mask_list.append(dwi_mask_sagittal)
    mask_list.append(dwi_mask_coronal)
    mask_list.append(dwi_mask_axial)

    mask_file = storage + '/' + "dwi_mask.txt"
    with open(mask_file, "w") as a:
        for item in mask_list:
            a.write(item + "\n")

    end_masking_time = datetime.datetime.now()
    total_masking_time = end_masking_time - start_total_time
    print ("Masking Time Taken : ", round(int(total_masking_time.seconds)/60, 2), " min")
