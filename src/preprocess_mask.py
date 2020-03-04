#!/usr/bin/env python
import os
from os import path
import sys
import numpy as np
import nibabel as nib
import argparse
import pathlib

def process_trainingdata(mask_arr):
    count = 0
    for b0_mask in mask_arr:
        img = nib.load(b0_mask)
        imgU8_sagittal = img.get_data().astype(np.uint8) # sagittal view
        imgU8_sagittal[ imgU8_sagittal < 0 ] = 0        
        imgU8_sagittal[ imgU8_sagittal > 1 ] = 1        
        imgU8_coronal = np.swapaxes(imgU8_sagittal,0,1) # coronal view
        imgU8_axial = np.swapaxes(imgU8_sagittal,0,2)   # Axial view

        # dwi mask volume data is written to the binary file
        imgU8_sagittal.tofile(sagittal_f_handle)
        imgU8_coronal.tofile(coronal_f_handle)
        imgU8_axial.tofile(axial_f_handle)

        print('Case ' + str(count) + ' done')
        count = count + 1

    # Closing the binary file
    sagittal_f_handle.close()
    axial_f_handle.close()
    coronal_f_handle.close()

# parser module for input arguments
SUFFIX_TXT = "txt"
parser = argparse.ArgumentParser()
parser.add_argument('-i', action='store', dest='dwi', type=str,
                        help=" input dwi masks file in txt format")
args = parser.parse_args()

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
            mask_arr = f.read().splitlines()


storage = path.dirname(mask_arr[0])

# dwi cases mask will be written to the below binary files
sagittal_bin_file = storage + '/sagittal-binary-mask'
coronal_bin_file = storage + '/coronal-binary-mask'
axial_bin_file = storage + '/axial-binary-mask'

# The above binary files will be converted to 3D numpy array
sagittal_trainingdata = storage + '/sagittal-traindata-mask.npy'
coronal_trainingdata = storage + '/coronal-traindata-mask.npy'
axial_trainingdata = storage + '/axial-traindata-mask.npy'

# Open the binary file for writing
sagittal_f_handle = open(sagittal_bin_file, 'wb')
coronal_f_handle = open(coronal_bin_file, 'wb')
axial_f_handle = open(axial_bin_file, 'wb')

process_trainingdata(mask_arr)

x_dim=len(mask_arr)*256
y_dim=256
z_dim=256

# Open the binary file and convert it to 3D numpy array
merge_sagittal = np.memmap(sagittal_bin_file, dtype=np.uint8, mode='r+', shape=(x_dim, y_dim, z_dim))
print("Saving sagittal training data mask to disk")
np.save(sagittal_trainingdata, merge_sagittal)
os.unlink(sagittal_bin_file)

merge_coronal = np.memmap(coronal_bin_file, dtype=np.uint8, mode='r+', shape=(x_dim, y_dim, z_dim))
print("Saving coronal training data mask to disk")
np.save(coronal_trainingdata, merge_coronal)
os.unlink(coronal_bin_file)

merge_axial = np.memmap(axial_bin_file, dtype=np.uint8, mode='r+', shape=(x_dim, y_dim, z_dim))
print("Saving axial training data mask to disk")
np.save(axial_trainingdata, merge_axial)
os.unlink(axial_bin_file)