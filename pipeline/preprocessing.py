from __future__ import division
# -----------------------------------------------------------------
# Author:       Senthil Palanivelu, Tashrif Billah                 
# Written:      01/22/2020                             
# Last Updated:     01/31/2020
# Purpose:          Pre-processing pipeline for diffusion brain masking
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
"""

# pylint: disable=invalid-name
import os
import os.path
from os import path
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
from multiprocessing import Process, Manager, Value, Pool
import multiprocessing as mp
import sys
from time import sleep
import scipy as sp
import scipy.misc, scipy.ndimage.interpolation
import os

# suffixes
SUFFIX_NIFTI = "nii"
SUFFIX_NIFTI_GZ = "nii.gz"
SUFFIX_NRRD = "nrrd"
SUFFIX_NHDR = "nhdr"
SUFFIX_NPY = "npy"
SUFFIX_TXT = "txt"
output_mask = []


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
    bashCommand1 = ("unu head " + input_file + " | grep -i sizes | awk '{print $5}'")
    bashCommand2 = ("unu head " + input_file + " | grep -i _gradient_ | wc -l")
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


if __name__ == '__main__':

    start_total_time = datetime.datetime.now()
    # parser module for input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', action='store', dest='dwi', type=str,
                        help=" input caselist file in txt format")

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
                reference_list = manager.list()
                omat_list = []                
                jobs = []

                lock = mp.Lock()
                for i in range(0,len(case_arr)):
                    p = mp.Process(target=pre_process, args=(lock,case_arr[i],
                                                             reference_list))
                    p.start()
                    jobs.append(p)
        
                for process in jobs:
                    process.join()

                reference_list = list(reference_list)

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

            merged_dwi_list = []
            merged_dwi_list.append(cases_file_s)
            merged_dwi_list.append(cases_file_c)
            merged_dwi_list.append(cases_file_a)

            merge_s = np.memmap(binary_file_s, dtype=np.float32, mode='r+', shape=(256 * len(reference_list), y_dim, z_dim))
            merge_c = np.memmap(binary_file_c, dtype=np.float32, mode='r+', shape=(256 * len(reference_list), y_dim, z_dim))
            merge_a = np.memmap(binary_file_a, dtype=np.float32, mode='r+', shape=(256 * len(reference_list), y_dim, z_dim))

            print ("Saving data to disk...")
            np.save(cases_file_s, merge_s)
            np.save(cases_file_c, merge_c)
            np.save(cases_file_a, merge_a)

            shuffled_file = storage + '/' + "shulled_cases.txt"
            merged_file = storage + '/' + "merged_cases.txt"
            registered_file = storage + '/' + "ants_cases.txt"
            mat_file = storage + '/' + "mat_cases.txt"
            reference_file = storage + '/' + "reference_cases.txt"

            with open(shuffled_file, "w") as dwi_file_axial:
                for item in shuffled_list:
                    dwi_file_axial.write(item + "\n")

            with open(merged_file, "w") as merged_dwi:
                for item in merged_dwi_list:
                    merged_dwi.write(item + "\n")

            with open(registered_file, "w") as reg_dwi:
                for item in transformed_cases:
                    reg_dwi.write(item + "\n")

            with open(mat_file, "w") as mat_dwi:
                for item in omat_list:
                    mat_dwi.write(item + "\n")
                    
            with open(reference_file, "w") as ref_dwi:
                for item in reference_list:
                    ref_dwi.write(item + "\n")

            end_preprocessing_time = datetime.datetime.now()
            total_preprocessing_time = end_preprocessing_time - start_total_time
            print ("Pre-Processing Time Taken : ", round(int(total_preprocessing_time.seconds)/60, 2), " min")
