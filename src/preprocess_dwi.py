import os
import sys
import numpy as np
import nibabel as nib
import argparse
import pathlib

# dwi cases will be written to the below binary files
sagittal_bin_file = 'sagittal-binary-dwi'
coronal_bin_file = 'coronal-binary-dwi'
axial_bin_file = 'axial-binary-dwi'

# The above binary files will be converted to 3D numpy array
sagittal_trainingdata = 'sagittal-traindata-dwi.npy'
coronal_trainingdata = 'coronal-traindata-dwi.npy'
axial_trainingdata = 'axial-traindata-dwi.npy'

# Open the binary file for writing
sagittal_f_handle = open(sagittal_bin_file, 'wb')
coronal_f_handle = open(coronal_bin_file, 'wb')
axial_f_handle = open(axial_bin_file, 'wb')

# Function to preprocess the dwi cases
def process_trainingdata(dwib0_arr):
    count = 0
    for b0 in dwib0_arr:
        img = nib.load(b0)
        imgF32 = img.get_data().astype(np.float32)
        ''' Intensity based segmentation of MR images is hampered by radio frerquency field
            inhomogeneity causing intensity variation. The intensity range is typically
            scaled between the highest and lowest signal in the Image. Intensity values
            of the same tissue can vary between scans. The pixel value in images must be
            scaled prior to providing the images as input to CNN. The data is projected in to
            a predefined range [0,1] '''
        p = np.percentile(imgF32, 99)
        imgF32_sagittal = imgF32 / p 			          # sagittal view
        imgF32_sagittal[ imgF32_sagittal < 0 ] =  sys.float_info.epsilon	      
        imgF32_sagittal[ imgF32_sagittal > 1 ] = 1 		  
        imgF32_coronal = np.swapaxes(imgF32_sagittal,0,1) # coronal view
        imgF32_axial = np.swapaxes(imgF32_sagittal,0,2)   # Axial view

        # dwi volume data is written to the binary file
        imgF32_sagittal.tofile(sagittal_f_handle)
        imgF32_coronal.tofile(coronal_f_handle)
        imgF32_axial.tofile(axial_f_handle)

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
                        help=" input dwi cases file in txt format")
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
            dwib0_arr = f.read().splitlines()

process_trainingdata(dwib0_arr)
x_dim=len(dwib0_arr)*256
y_dim=256
z_dim=256

# Open the binary file and convert it to 3D numpy array
merge_sagittal = np.memmap(sagittal_bin_file, dtype=np.float32, mode='r+', shape=(x_dim, y_dim, z_dim))
print("Saving sagittal training data to disk")
np.save(sagittal_trainingdata, merge_sagittal)
os.unlink(sagittal_bin_file)

merge_coronal = np.memmap(coronal_bin_file, dtype=np.float32, mode='r+', shape=(x_dim, y_dim, z_dim))
print("Saving coronal training data to disk")
np.save(coronal_trainingdata, merge_coronal)
os.unlink(coronal_bin_file)

merge_axial = np.memmap(axial_bin_file, dtype=np.float32, mode='r+', shape=(x_dim, y_dim, z_dim))
print("Saving axial training data to disk")
np.save(axial_trainingdata, merge_axial)
os.unlink(axial_bin_file)
