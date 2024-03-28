#!/usr/bin/env python

from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure
from skimage.measure import label, regionprops
import numpy as np
import nibabel as nib
import sys
from os.path import abspath

struct_element= generate_binary_structure(3,1)

def findLargestConnectMask(img):

    mask = label(img > 0, connectivity=1)
    maxArea = 0
    for region in regionprops(mask):
        if region.area > maxArea:
            maxLabel = region.label
            maxArea = region.area

    largeConnectMask = (mask == maxLabel)

    return largeConnectMask


    
def single_pass(InputImage, scale):

    temp_image= InputImage.copy()

    for s in range(scale,0,-1):
        temp_image= single_scale(temp_image, s)

    temp_image= findLargestConnectMask(temp_image)

    # OutputImage
    return temp_image



def single_scale(InputImage, ss):

    # erosion
    del_image= binary_erosion(InputImage, struct_element, iterations= ss)

    # largest connected component
    largest_image= findLargestConnectMask(del_image)

    del_image[largest_image>0]= 0

    # dilation
    del_image= binary_dilation(del_image, struct_element, iterations= ss+1)

    ind_to_delete= del_image>0
    ind_to_keep= ~ind_to_delete
    largest_image[ind_to_delete]= 0
    largest_image[ind_to_keep]= InputImage[ind_to_keep]

    # OutputImage
    return largest_image


def perform_morph(temp_in, scale):

    temp_out= single_pass(temp_in, scale)
    while (temp_in!=temp_out).any():
        temp_in= temp_out.copy()
        temp_out= single_pass(temp_in, scale)

    return temp_out


def maskfilter(maskPath, scale, filtered_maskPath):
    '''
    This python executable replicates the functionality of
    https://github.com/MRtrix3/mrtrix3/blob/master/core/filter/mask_clean.h
    It performs a few erosion and dilation to remove islands of non-brain region in a brain mask.
    '''

    mask= nib.load(maskPath)

    filtered_mask= perform_morph(mask.get_fdata(), scale)

    nib.Nifti1Image(filtered_mask, affine= mask.affine, header= mask.header).to_filename(filtered_maskPath)



if __name__=='__main__':


    if len(sys.argv)==1 or sys.argv[1]=='-h' or sys.argv[1]=='--help':
        print('''This python executable replicates the functionality of 
https://github.com/MRtrix3/mrtrix3/blob/master/core/filter/mask_clean.h
It performs a few erosion and dilation to remove islands of non-brain region in a brain mask.   

Usage: maskfilter input scale output

See https://github.com/MRtrix3/mrtrix3/blob/master/core/filter/mask_clean.h for details''')
        exit()

    maskPath= abspath(sys.argv[1])
    scale= int(sys.argv[2])
    filtered_maskPath= abspath(sys.argv[3])

    maskfilter(maskPath, scale, filtered_maskPath)

