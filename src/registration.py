#!/usr/bin/env python
import os
from multiprocessing import Process, Manager, Value, Pool
import multiprocessing as mp
import subprocess
import argparse
import pathlib
import sys


def ANTS_rigid_body_trans(b0_nii, result, mask_file, reference):

    print("Performing ants rigid body transformation...")
    input_file = b0_nii
    case_name = os.path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-'
    output_file = os.path.join(os.path.dirname(input_file), output_name)

    trans_matrix = "antsRegistrationSyNQuick.sh -d 3 -f " + reference + " -m " + input_file + " -t r -o " + output_file
    output1 = subprocess.check_output(trans_matrix, shell=True)

    omat_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-0GenericAffine.mat'
    omat_file = os.path.join(os.path.dirname(input_file), omat_name)

    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-Warped.nii.gz'
    transformed_file = os.path.join(os.path.dirname(input_file), output_name)

    result.append((transformed_file, omat_file, mask_file))

SUFFIX_TXT = "txt"
SUFFIX_NIFTI_GZ = "nii.gz"
parser = argparse.ArgumentParser()
parser.add_argument('-b0', action='store', dest='b0', type=str,
                        help="txt file containing list of /path/to/b0, one path in each line")
parser.add_argument('-mask', action='store', dest='mask', type=str,
                        help="txt file containing list of /path/to/mask, one path in each line")
parser.add_argument('-ref', action='store', dest='ref', type=str,
                        help="reference b0 file for registration")

args = parser.parse_args()
reference = str(args.ref)

try:
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        parser.error('too few arguments')
        sys.exit(0)

except SystemExit:
    sys.exit(0)

if args.b0:
    f = pathlib.Path(args.b0)
    if f.exists():
        print ("File exist")
        filename = args.b0
    else:
        print ("File not found")
        sys.exit(1)

    # Input caselist.txt
    if filename.endswith(SUFFIX_TXT):
        with open(filename) as f:
            target_list = f.read().splitlines()

if args.mask:
    f = pathlib.Path(args.mask)
    if f.exists():
        print ("File exist")
        filename = args.mask
    else:
        print ("File not found")
        sys.exit(1)

    # Input caselist.txt
    if filename.endswith(SUFFIX_TXT):
        with open(filename) as f:
            mask_list = f.read().splitlines()

with Manager() as manager:
    result = manager.list()              
    ants_jobs = []
    for i in range(0, len(target_list)):
        p_ants = mp.Process(target=ANTS_rigid_body_trans, args=(target_list[i],
                                                             result, mask_list[i], reference))
        ants_jobs.append(p_ants)
        p_ants.start()
        
    for process in ants_jobs:
        process.join()

    result = list(result)

transformed_cases = []
omat_list = []
masks_new_list = []

for subject_ANTS in result:
    transformed_cases.append(subject_ANTS[0])
    omat_list.append(subject_ANTS[1])
    masks_new_list.append(subject_ANTS[2])

# Apply the same tranformation to the mask file
for i in range(0, len(transformed_cases)):

    input_file = transformed_cases[i]
    case_name = os.path.basename(input_file)
    output_name = case_name[:len(case_name) - (len(SUFFIX_NIFTI_GZ) + 1)] + '-mask.nii.gz'
    output_file = os.path.join(os.path.dirname(masks_new_list[i]), output_name)
    apply_mask_trans = "antsApplyTransforms -d 3 -i " + masks_new_list[i] + " -r " + input_file + " -o " \
                            + output_file + " --transform [" + omat_list[i] + "]"

    output2 = subprocess.check_output(apply_mask_trans, shell=True)
