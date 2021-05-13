#!/usr/bin/env python

from __future__ import division
# -----------------------------------------------------------------
# Author:       Senthil Palanivelu, Tashrif Billah                 
# Written:      01/22/2020                             
# Last Updated:     02/28/2020
# Purpose:          Pipeline for diffusion brain masking
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
10)  Applys Inverse tranformation
10)  Cleaning
"""


# pylint: disable=invalid-name
import os
from os import path
import webbrowser
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

    case_name = path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NHDR) + 1)] + '-multi-mask.npy'
    output_file = path.join(path.dirname(input_file), output_name)

    SO = multi_view.astype('float32')
    np.save(output_file, SO)
    return output_file


def normalize(b0_resampled, percentile, data_n):
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
    case_name = path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-normalized.nii.gz'
    output_file = path.join(path.dirname(input_file), output_name)
    img = nib.load(b0_resampled)
    imgU16 = img.get_data().astype(np.float32)
    p = np.percentile(imgU16, percentile)
    data = imgU16 / p
    data[data > 1] = 1
    data[data < 0] = 0
    image_dwi = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(image_dwi, output_file)
    data_n.append(output_file)


def save_nifti(fname, data, affine=None, hdr=None):
   
    hdr.set_data_dtype('int16')
    result_img = nib.Nifti1Image(data, affine, header=hdr)
    result_img.to_filename(fname)


def npy_to_nifti(b0_normalized_cases, cases_mask_arr, sub_name, view='default', reference='default', omat=None):
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
        output_dir = path.dirname(sub_name[i])
        output_file = cases_mask_arr[i][:len(cases_mask_arr[i]) - len(SUFFIX_NPY)] + 'nii.gz'
        nib.save(image_predict, output_file)

        output_file_inverseMask = ANTS_inverse_transform(output_file, reference[i], omat[i])
        Ants_inverse_output_file = output_file_inverseMask

        case_name = path.basename(Ants_inverse_output_file)
        fill_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-filled.nii.gz'
        filled_file = path.join(output_dir, fill_name)
        fill_cmd = "ImageMath 3 " + filled_file + " FillHoles " + Ants_inverse_output_file
        process = subprocess.Popen(fill_cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        subject_name = path.basename(sub_name[i])
        if subject_name.endswith(SUFFIX_NIFTI_GZ):
            format = SUFFIX_NIFTI_GZ
        else:
            format = SUFFIX_NIFTI

        # Neural Network Predicted Mask
        CNN_predict_file = subject_name[:len(subject_name) - (len(format) + 1)] + '-' + view + '_originalMask.nii.gz'
        CNN_output_file = path.join(output_dir, CNN_predict_file)
        bashCommand = 'cp ' + filled_file + " " + CNN_output_file
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        output_filter_file = subject_name[:len(subject_name) - (len(format) + 1)] + '-' + view + '_FilteredMask.nii.gz'
        output_mask_filtered = path.join(output_dir, output_filter_file)

        if args.filter:
            print('Cleaning up ', CNN_output_file)
            
            if args.filter=='mrtrix':
                mask_filter = "maskfilter -force " + CNN_output_file + " -scale 2 clean " + output_mask_filtered

            elif args.filter=='scipy':
                mask_filter = path.join(path.dirname(__file__),'../src/maskfilter.py') + f' {CNN_output_file} 2 {output_mask_filtered}'
        
            process = subprocess.Popen(mask_filter.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

        else:
            output_mask_filtered= CNN_output_file


        print(output_mask_filtered)
        img = nib.load(output_mask_filtered)
        data_dwi = nib.load(sub_name[i])
        imgU16 = img.get_data().astype(np.uint8)

        brain_mask_file = subject_name[:len(subject_name) - (len(format) + 1)] + '-' + view + '_BrainMask.nii.gz'
        brain_mask_final = path.join(output_dir, brain_mask_file)

        save_nifti(brain_mask_final, imgU16, affine=data_dwi.affine, hdr=data_dwi.header)
        output_mask.append(brain_mask_final)

    return output_mask


def clear(directory):
    print ("Cleaning files ...")

    bin_a = 'cases_' + str(os.getpid()) + '_binary_a'
    bin_s = 'cases_' + str(os.getpid()) + '_binary_s'
    bin_c = 'cases_' + str(os.getpid()) + '_binary_c'

    for filename in os.listdir(directory):
        if filename.startswith('Comp') | filename.endswith(SUFFIX_NPY) | \
                filename.endswith('_SO.nii.gz') | filename.endswith('downsampled.nii.gz') | \
                filename.endswith('-thresholded.nii.gz') | filename.endswith('-inverse.mat') | \
                filename.endswith('-Warped.nii.gz') | filename.endswith('-0GenericAffine.mat') | \
                filename.endswith('_affinedMask.nii.gz') | filename.endswith('_originalMask.nii.gz') | \
                filename.endswith('multi-mask.nii.gz') | filename.endswith('-mask-inverse.nii.gz') |  \
                filename.endswith('-InverseWarped.nii.gz') | filename.endswith('-FilteredMask.nii.gz') | \
                filename.endswith(bin_a) | filename.endswith(bin_c) | filename.endswith(bin_s) | \
                filename.endswith('_FilteredMask.nii.gz') | filename.endswith('-normalized.nii.gz') | filename.endswith('-filled.nii.gz'):
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


def ANTS_rigid_body_trans(b0_nii, result, reference=None):

    print("Performing ants rigid body transformation...")
    input_file = b0_nii
    case_name = path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-'
    output_file = path.join(path.dirname(input_file), output_name)

    trans_matrix = "antsRegistrationSyNQuick.sh -d 3 -f " + reference + " -m " + input_file + " -t r -o " + output_file
    output1 = subprocess.check_output(trans_matrix, shell=True)

    omat_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-0GenericAffine.mat'
    omat_file = path.join(path.dirname(input_file), omat_name)

    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-Warped.nii.gz'
    transformed_file = path.join(path.dirname(input_file), output_name)

    result.append((transformed_file, omat_file))


def ANTS_inverse_transform(predicted_mask, reference, omat='default'):

    #print "Mask file = ", predicted_mask
    #print "Reference = ", reference
    #print "omat = ", omat

    print("Performing ants inverse transform...")
    input_file = predicted_mask
    case_name = path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-inverse.nii.gz'
    output_file = path.join(path.dirname(input_file), output_name)

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


def pre_process(input_file, target_list, b0_threshold=50.):

    from conversion import nifti_write, read_bvals
    from subprocess import Popen

    if path.isfile(input_file):

        # convert NRRD/NHDR to NIFIT as the first step
        # extract bse.py from just NIFTI later
        if input_file.endswith(SUFFIX_NRRD) | input_file.endswith(SUFFIX_NHDR):
            inPrefix= input_file.split('.')[0]
            nifti_write(input_file)
            input_file= inPrefix+ '.nii.gz'

        inPrefix= input_file.split('.nii')[0]
        b0_nii= path.join(inPrefix+ '_bse.nii.gz')
        
        dwi= nib.load(input_file)

        if len(dwi.shape)>3:
            print("Extracting b0 volume...")
            bvals= np.array(read_bvals(input_file.split('.nii')[0]+ '.bval'))
            where_b0= np.where(bvals <= b0_threshold)[0]
            b0= dwi.get_data()[...,where_b0].mean(-1)
        else:
            print("Loading b0 volume...")
            b0= dwi.get_fdata()

        np.nan_to_num(b0).clip(min= 0., out= b0)
        nib.Nifti1Image(b0, affine= dwi.affine, header= dwi.header).to_filename(b0_nii)

        target_list.append((b0_nii))

    else:
        print("File not found ", input_file)
        sys.exit(1)


def remove_string(input_file, output_file, string):
    infile = input_file
    outfile = output_file
    delete_list = [string]
    fin = open(infile)
    fout = open(outfile, "w+")
    for line in fin:
        for word in delete_list:
            line = line.replace(word, "")
        fout.write(line)
    fin.close()
    fout.close()


def quality_control(mask_list, target_list, tmp_path, view='default'):
    '''The slicesdir command takes the list of images and creates a simple web-page containing snapshots for each of the images.
    Once it has finished running it tells you the name of the web page to open in your web browser, to view the snapshots.
    '''

    slices = " "
    for i in range(0, len(mask_list)):
        str1 = target_list[i]
        str2 = mask_list[i]
        slices += str1 + " " + str2 + " "

    final = "slicesdir -o" + slices
    dir_bak = os.getcwd()
    os.chdir(tmp_path)

    process= subprocess.Popen(final, shell=True)
    process.wait()
    os.chdir(dir_bak)

    mask_folder = os.path.join(tmp_path, 'slicesdir')
    mask_newfolder = os.path.join(tmp_path, 'slicesdir_' + view)
    if os.path.exists(mask_newfolder):
        process = subprocess.Popen('rm -rf '+ mask_newfolder, shell=True)
        process.wait()

    process = subprocess.Popen('mv ' + mask_folder + " " + mask_newfolder, shell=True)
    process.wait()
    

if __name__ == '__main__':

    start_total_time = datetime.datetime.now()
    # parser module for input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', action='store', dest='dwi', type=str,
                        help="txt file containing list of /path/to/dwi, one path in each line")

    parser.add_argument('-f', action='store', dest='model_folder', type=str,
                        help="folder containing the trained models")

    parser.add_argument("-a", type=str2bool, dest='Axial', nargs='?',
                        const=True, default=False,
                        help="advanced option to generate multiview and axial Mask (yes/true/y/1)")

    parser.add_argument("-c", type=str2bool, dest='Coronal', nargs='?',
                        const=True, default=False,
                        help="advanced option to generate multiview and coronal Mask (yes/true/y/1)")

    parser.add_argument("-s", type=str2bool, dest='Sagittal', nargs='?',
                        const=True, default=False,
                        help="advanced option to generate multiview and sagittal Mask (yes/true/y/1)")

    parser.add_argument("-qc", type=str2bool, dest='snap', nargs='?',
                        const=True, default=False,
                        help="open snapshots in your web browser (yes/true/y/1)")

    parser.add_argument('-p', type=int, dest='percentile', default=99, help='Percentile to normalize Image [0, 1]')

    parser.add_argument('-nproc', type=int, dest='cr', default=8, help='number of processes to use')
    
    parser.add_argument('-filter', choices=['scipy','mrtrix'], help='''perform morphological operation on the 
CNN generated mask to clean up holes and islands, can be done through a provided script (scipy) 
or MRtrix3 maskfilter (mrtrix)''')

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
            reference = trained_model_folder + '/IITmean_b0_256.nii.gz'

            binary_file_s = storage + '/' + unique + '_' + str(os.getpid()) + '_binary_s'
            binary_file_c = storage + '/' + unique + '_' + str(os.getpid()) + '_binary_c'
            binary_file_a = storage + '/' + unique + '_' + str(os.getpid()) + '_binary_a'

            f_handle_s = open(binary_file_s, 'wb')
            f_handle_c = open(binary_file_c, 'wb')
            f_handle_a = open(binary_file_a, 'wb')

            x_dim = 0
            y_dim = 256
            z_dim = 256
            transformed_cases = []
            """
            Enable Multi core Processing for pre processing
            manager provide a way to create data which can be shared between different processes
            """
            with Manager() as manager:
                target_list = manager.list()
                omat_list = []                
                jobs = []
                for i in range(0,len(case_arr)):
                    p_process = mp.Process(target=pre_process, args=(case_arr[i],
                                                             target_list))
                    p_process.start()
                    jobs.append(p_process)
        
                for process in jobs:
                    process.join()

                target_list = list(target_list)
            """
            Enable Multi core Processing for ANTS Registration
            manager provide a way to create data which can be shared between different processes
            """
            with Manager() as manager:
                result = manager.list()              
                ants_jobs = []
                for i in range(0, len(target_list)):
                    p_ants = mp.Process(target=ANTS_rigid_body_trans, args=(target_list[i],
                                                             result, reference))
                    ants_jobs.append(p_ants)
                    p_ants.start()
        
                for process in ants_jobs:
                    process.join()

                result = list(result)

            for subject_ANTS in result:
                transformed_cases.append(subject_ANTS[0])
                omat_list.append(subject_ANTS[1])

            with Manager() as manager:
                data_n = manager.list() 
                norm_jobs = []             
                for i in range(0, len(target_list)):
                    p_norm = mp.Process(target=normalize, args=(transformed_cases[i],
                                                             args.percentile, data_n))
                    norm_jobs.append(p_norm)
                    p_norm.start()
        
                for process in norm_jobs:
                    process.join()

                data_n = list(data_n)

            
            count = 0
            for b0_nifti in data_n:
                img = nib.load(b0_nifti)
                imgU16_sagittal = img.get_data().astype(np.float32)  # sagittal view
                imgU16_coronal = np.swapaxes(imgU16_sagittal, 0, 1)  # coronal view
                imgU16_axial = np.swapaxes(imgU16_sagittal, 0, 2)    # Axial view

                imgU16_sagittal.tofile(f_handle_s)
                imgU16_coronal.tofile(f_handle_c)
                imgU16_axial.tofile(f_handle_a)

                print ("Case completed = ", count)
                count += 1

            f_handle_s.close()
            f_handle_c.close()
            f_handle_a.close()

            print ("Merging npy files...")
            cases_file_s = storage + '/'+ unique + '_' + str(os.getpid()) + '-casefile-sagittal.npy'
            cases_file_c = storage + '/'+ unique + '_' + str(os.getpid()) + '-casefile-coronal.npy'
            cases_file_a = storage + '/'+ unique + '_' + str(os.getpid()) + '-casefile-axial.npy'

            merged_dwi_list = []
            merged_dwi_list.append(cases_file_s)
            merged_dwi_list.append(cases_file_c)
            merged_dwi_list.append(cases_file_a)

            merge_s = np.memmap(binary_file_s, dtype=np.float32, mode='r+', shape=(256 * len(target_list), y_dim, z_dim))
            merge_c = np.memmap(binary_file_c, dtype=np.float32, mode='r+', shape=(256 * len(target_list), y_dim, z_dim))
            merge_a = np.memmap(binary_file_a, dtype=np.float32, mode='r+', shape=(256 * len(target_list), y_dim, z_dim))

            print ("Saving data to disk...")
            np.save(cases_file_s, merge_s)
            np.save(cases_file_c, merge_c)
            np.save(cases_file_a, merge_a)

            normalized_file = storage + "/norm_cases_" + str(os.getpid()) + ".txt"
            registered_file = storage + "/ants_cases_" + str(os.getpid()) + ".txt"
            mat_file = storage + "/mat_cases_" + str(os.getpid()) + ".txt"
            target_file = storage + "/target_cases_" + str(os.getpid()) + ".txt"

            with open(normalized_file, "w") as norm_dwi:
                for item in data_n:
                    norm_dwi.write(item + "\n")

            remove_string(normalized_file, registered_file, "-normalized")
            remove_string(registered_file, target_file, "-Warped")

            with open(target_file) as f:
                newText=f.read().replace('.nii.gz', '-0GenericAffine.mat')

            with open(mat_file, "w") as f:
                f.write(newText)

            end_preprocessing_time = datetime.datetime.now()
            total_preprocessing_time = end_preprocessing_time - start_total_time
            print ("Pre-Processing Time Taken : ", round(int(total_preprocessing_time.seconds)/60, 2), " min")

            # DWI Deep Learning Segmentation
            dwi_mask_sagittal = predict_mask(cases_file_s, trained_model_folder, view='sagittal')
            dwi_mask_coronal = predict_mask(cases_file_c, trained_model_folder, view='coronal')
            dwi_mask_axial = predict_mask(cases_file_a, trained_model_folder, view='axial')

            end_masking_time = datetime.datetime.now()
            total_masking_time = end_masking_time - start_total_time - total_preprocessing_time
            print ("Masking Time Taken : ", round(int(total_masking_time.seconds)/60, 2), " min")

            transformed_file = registered_file
            omat_file = mat_file

            transformed_cases = [line.rstrip('\n') for line in open(transformed_file)]
            target_list = [line.rstrip('\n') for line in open(target_file)]
            omat_list = [line.rstrip('\n') for line in open(omat_file)]

            # Post Processing
            print ("Splitting files....")
            cases_mask_sagittal = split(dwi_mask_sagittal, target_list, view='sagittal')
            cases_mask_coronal = split(dwi_mask_coronal, target_list, view='coronal')
            cases_mask_axial = split(dwi_mask_axial, target_list, view='axial')

            multi_mask = []
            for i in range(0, len(cases_mask_sagittal)):

                sagittal_SO = cases_mask_sagittal[i]
                coronal_SO = cases_mask_coronal[i]
                axial_SO = cases_mask_axial[i]

                input_file = target_list[i]

                multi_view_mask = multi_view_fast(sagittal_SO, 
                                                  coronal_SO, 
                                                  axial_SO, 
                                                  input_file)


                brain_mask_multi = npy_to_nifti(list(transformed_cases[i].split()), 
                                                list(multi_view_mask.split()), 
                                                list(target_list[i].split()),
                                                view='multi', 
                                                reference=list(target_list[i].split()), 
                                                omat=list(omat_list[i].split()))


                print ("Mask file : ", brain_mask_multi)
                multi_mask.append(brain_mask_multi[0])
            quality_control(multi_mask, target_list, tmp_path, view='multi')

            if args.Sagittal:
                omat = omat_list
            else:
                omat = None

            if args.Sagittal:
                sagittal_mask = npy_to_nifti(transformed_cases, 
                                            cases_mask_sagittal, 
                                            target_list,
                                            view='sagittal', 
                                            reference=target_list,
                                            omat=omat)
                list_masks(sagittal_mask, view='sagittal')
                quality_control(sagittal_mask, target_list, tmp_path, view='sagittal')

            if args.Coronal:
                omat = omat_list
            else:
                omat = None

            if args.Coronal:
                coronal_mask = npy_to_nifti(transformed_cases, 
                                          cases_mask_coronal, 
                                          target_list,
                                          view='coronal', 
                                          reference=target_list,
                                          omat=omat)
                list_masks(coronal_mask, view='coronal')
                quality_control(coronal_mask, target_list, tmp_path, view='coronal')

            if args.Axial:
                omat = omat_list
            else:
                omat = None

            if args.Axial:
                axial_mask = npy_to_nifti(transformed_cases, 
                                         cases_mask_axial, 
                                         target_list,
                                         view='axial', 
                                         reference=target_list,
                                         omat=omat)
                list_masks(axial_mask, view='axial')
                quality_control(axial_mask, target_list, tmp_path, view='axial')

            for i in range(0, len(cases_mask_sagittal)):
                clear(path.dirname(cases_mask_sagittal[i]))

            if args.snap:
                webbrowser.open(path.join(tmp_path, 'slicesdir_multi/index.html'))
                if args.Sagittal:
                    webbrowser.open(path.join(tmp_path, 'slicesdir_sagittal/index.html'))
                if args.Coronal:
                    webbrowser.open(path.join(tmp_path, 'slicesdir_coronal/index.html'))
                if args.Axial:
                    webbrowser.open(path.join(tmp_path, 'slicesdir_axial/index.html'))

        end_total_time = datetime.datetime.now()
        total_t = end_total_time - start_total_time
        print ("Total Time Taken : ", round(int(total_t.seconds)/60, 2), " min")

