#!/usr/bin/env python

from __future__ import division

"""
pipeline.py
~~~~~~~~~~
01)  Performs Multi View Aggregation
02)  Applys Inverse tranformation
03)  Cleaning
"""

import os
import webbrowser
from os import path
import re
import sys
import subprocess
import argparse, textwrap
import datetime
import pathlib
import nibabel as nib
import numpy as np
from time import sleep

# suffixes
SUFFIX_NIFTI = "nii"
SUFFIX_NIFTI_GZ = "nii.gz"
SUFFIX_NRRD = "nrrd"
SUFFIX_NHDR = "nhdr"
SUFFIX_NPY = "npy"
SUFFIX_TXT = "txt"
output_mask = []


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

        #print(CNN_output_file)
        img = nib.load(CNN_output_file)
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

            process_file = storage + "/process_id.txt"
            with open(process_file) as pf:
                process_id_arr = pf.read().splitlines()

            masked_file = storage + "/b0_NPYmasks_" + process_id_arr[-1] + ".txt"
            with open(masked_file) as f:
                masked_cases_npy = f.read().splitlines()
         
            dwi_mask_sagittal = masked_cases_npy[0]
            dwi_mask_coronal = masked_cases_npy[1]
            dwi_mask_axial = masked_cases_npy[2]

            transformed_file = storage + "/ants_cases_" + process_id_arr[-1] + ".txt"
            target_file = storage + "/target_cases_" + process_id_arr[-1] + ".txt"
            omat_file = storage + "/mat_cases_" + process_id_arr[-1] + ".txt"

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
        print ("Post Processing Time Taken : ", round(int(total_t.seconds)/60, 2), " min")

