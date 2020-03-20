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

        print("Extracting b0 volume...")
     
        dwi= nib.load(input_file)
        bvals= np.array(read_bvals(input_file.split('.nii')[0]+ '.bval'))
        where_b0= np.where(bvals <= b0_threshold)[0]
        b0= dwi.get_data()[...,where_b0].mean(-1)
        np.nan_to_num(b0).clip(min= 0., out= b0)
        nib.Nifti1Image(b0, affine= dwi.affine, header= dwi.header).to_filename(b0_nii)

        target_list.append((b0_nii))

    else:
        print("File not found ", input_file)
        sys.exit(1)


if __name__ == '__main__':

    start_total_time = datetime.datetime.now()
    # parser module for input arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', action='store', dest='dwi', type=str,
                        help="txt file containing list of /path/to/dwi, one path in each line")

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

            process_file = storage + "/process_id.txt"
            target_file = storage + "/b0_cases_" + str(os.getpid()) + ".txt"

            with open(target_file, "w") as b0_nii:
                for item in target_list:
                    b0_nii.write(item + "\n")
            
            with open(process_file, "a") as myfile:
                    myfile.write(str(os.getpid()) + "\n")

            end_preprocessing_time = datetime.datetime.now()
            total_preprocessing_time = end_preprocessing_time - start_total_time
            print ("Extract b0 Time Taken : ", round(int(total_preprocessing_time.seconds)/60, 2), " min")