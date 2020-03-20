#!/usr/bin/env python

from __future__ import division

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

import os
from os import path
import multiprocessing as mp
import re
import sys
import subprocess
import argparse, textwrap
import datetime
import pathlib
import nibabel as nib
import numpy as np
from multiprocessing import Process, Manager, Value, Pool
from time import sleep

# suffixes
SUFFIX_NIFTI = "nii"
SUFFIX_NIFTI_GZ = "nii.gz"
SUFFIX_NRRD = "nrrd"
SUFFIX_NHDR = "nhdr"
SUFFIX_NPY = "npy"
SUFFIX_TXT = "txt"
output_mask = []

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


if __name__ == '__main__':

    start_total_time = datetime.datetime.now()
    # parser module for input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', action='store', dest='dwi', type=str,
                        help="txt file containing list of /path/to/dwi, one path in each line")

    parser.add_argument('-f', action='store', dest='model_folder', type=str,
                        help="folder containing the trained models")

    parser.add_argument('-p', type=int, dest='percentile', default=99, help='Percentile to normalize Image [0, 1]')

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
            omat_list = []

            process_file = storage + "/process_id.txt"
            with open(process_file) as pf:
                process_id_arr = pf.read().splitlines()

            b0_files = storage + "/b0_cases_" + process_id_arr[-1] + ".txt"
            with open(b0_files) as f:
                target_list = f.read().splitlines()

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
            process_file = storage + "/process_id.txt"
            merged_file = storage + "/merged_cases_" + str(os.getpid()) + ".txt"

            with open(normalized_file, "w") as norm_dwi:
                for item in data_n:
                    norm_dwi.write(item + "\n")

            with open(merged_file, "w") as merged_b0:
                for item in merged_dwi_list:
                    merged_b0.write(item + "\n")

            with open(process_file, "a") as myfile:
                    myfile.write(str(os.getpid()) + "\n")

            remove_string(normalized_file, registered_file, "-normalized")
            remove_string(registered_file, target_file, "-Warped")

            with open(target_file) as f:
                newText=f.read().replace('.nii.gz', '-0GenericAffine.mat')

            with open(mat_file, "w") as f:
                f.write(newText)

            end_preprocessing_time = datetime.datetime.now()
            total_preprocessing_time = end_preprocessing_time - start_total_time
            print ("Pre-Processing Time Taken : ", round(int(total_preprocessing_time.seconds)/60, 2), " min")