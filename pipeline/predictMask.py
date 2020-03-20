#!/usr/bin/env python

from __future__ import division

"""
pipeline.py
~~~~~~~~~~
01)  Neural network brain mask prediction across the 3 principal axis

"""
import os
from os import path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress tensor flow message


# Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Get the first available GPU
try:
    import GPUtil
    DEVICE_ID_LIST = GPUtil.getFirstAvailable()
    DEVICE_ID = DEVICE_ID_LIST[0] # Grab first element from list
    print ("GPU found...", DEVICE_ID)

    # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

except:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("GPU not available...")

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

import multiprocessing as mp
import re
import sys
from glob import glob
import subprocess
import argparse, textwrap
import datetime
import pathlib
import nibabel as nib
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
from multiprocessing import Process, Manager, Value, Pool
from time import sleep
import keras
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


def predict_mask(input_file, trained_folder, view='default'):
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

    # load weights into new model
    optimal_model= glob(trained_folder + '/weights-' + view + '-improvement-*.h5')[-1]
    loaded_model.load_weights(optimal_model)

    # evaluate loaded model on test data
    loaded_model.compile(optimizer=Adam(lr=1e-5),
                         loss={'final_op': dice_coef_loss,
                               'xfinal_op': neg_dice_coef_loss,
                               'res_1_final_op': 'mse'})

    case_name = path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-' + view + '-mask.npy'
    output_file = path.join(path.dirname(input_file), output_name)

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
                        help="txt file containing list of /path/to/dwi, one path in each line")

    parser.add_argument('-f', action='store', dest='model_folder', type=str,
                        help="folder containing the trained models")

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


            TXT_file = path.basename(filename)
            #print(TXT_file)
            unique = TXT_file[:len(TXT_file) - (len(SUFFIX_TXT)+1)]
            #print(unique)
            storage = path.dirname(case_arr[0])
            tmp_path = storage + '/'
            trained_model_folder = args.model_folder.rstrip('/')

            process_file = storage + "/process_id.txt"
            with open(process_file) as pf:
                process_id_arr = pf.read().splitlines()

            merged_file = storage + "/merged_cases_" + process_id_arr[-1] + ".txt"
            with open(merged_file) as f:
                merged_cases_npy = f.read().splitlines()

            # DWI Deep Learning Segmentation
            mask_list = []
            dwi_mask_sagittal = predict_mask(merged_cases_npy[0], trained_model_folder, view='sagittal')
            dwi_mask_coronal = predict_mask(merged_cases_npy[1], trained_model_folder, view='coronal')
            dwi_mask_axial = predict_mask(merged_cases_npy[2], trained_model_folder, view='axial')

            mask_list.append(dwi_mask_sagittal)
            mask_list.append(dwi_mask_coronal)
            mask_list.append(dwi_mask_axial)

            mask_file = storage + "/b0_NPYmasks_" + process_id_arr[-1] + ".txt"
            with open(mask_file, "w") as a:
                for item in mask_list:
                    a.write(item + "\n")

            end_masking_time = datetime.datetime.now()
            total_masking_time = end_masking_time - start_total_time
            print ("Masking Time Taken : ", round(int(total_masking_time.seconds)/60, 2), " min")  
