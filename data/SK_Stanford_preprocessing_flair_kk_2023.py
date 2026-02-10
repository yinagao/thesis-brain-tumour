#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import nibabel as nib
from subprocess import call
from nipype.interfaces.ants import N4BiasFieldCorrection
import sys
from ast import literal_eval
import subprocess
import matplotlib.pyplot as plt
from skimage import exposure
import SimpleITK as sitk

# Paths for MRIs and tumor segmentations
# data_dir = r"C:\Users\karee\Documents\pLGG_EN_Nov2023\SK" # Location of FLAIR sequence and tumor segmentations
# data_dir = r"C:\Users\kareem kudus\Documents\pLGG_EN_Nov2023\SK" # Location of FLAIR sequence and tumor segmentations
data_dir1 = "C:/Users/Yina Gao/Documents/thesis-brain-tumour/data_output/dipg_before_preprocessing" # Location of FLAIR sequence and tumor segmentations
# data_dir2 = r"Z:\Datasets\MedicalImages\BrainData\SickKids\pLGG_EN_Nov2023\Stanford_EN" # Location of FLAIR sequence and tumor segmentations
atlas = r"Z:\Projects\SickKids_Brain_Preprocessing\SK_preprocessing_components\sri24_atlas\templates\T1.nii" # The template we are registering to
output_path = "C:/Users/Yina Gao/Documents/thesis-brain-tumour/data_output/dipg_preprocessed"

# Path to 3DSLICER (used for reorienting, registering, resampling volumes)
slicer_dir = "C:/Users/Yina Gao/AppData/Local/slicer.org/3D Slicer 5.10.0/bin/PythonSlicer.exe"

# Path to tool used for skull stripping
# bet_path = r'C:\Users\research\Documents\HD-BET\HD_BET\entry_point.py'
bet_path = "C:/Users/Yina Gao/Documents/thesis-brain-tumour/.venv/Lib/site-packages/HD_BET/entry_point.py"


# Settings for reorientation and registration
img_orientation = "RAI"
spacing = '1,1,1'
transform_type = 'Rigid,ScaleVersor3D,ScaleSkewVersor3D,Affine'
transform_mode = "Off"
sampling_percentage = 0.2

# Settings for interpolation...need to determine whether or not the below are optimal
interpolation_mode_resample = 'linear'
interpolation_mode_register_image = 'Linear'
interpolation_mode_register_segmentation = 'nn'


# NOTE FROM JAY:
########################
# HD-BET relies on nnunetv2 package but it's outdated in that the torch.load function needs to have weights_only=False. You have to go into the package code and update that
########################

# True if we want to skip preprocessing sequences that have already been preprocessed previously
skip_if_already_done = True

# This function produces a dictonary that contains the location of the FLAIR image and tumor segmentation for each patient that has it available
# This code was slightly adopted to work with multiple data directories
def find_flair_and_seg_files(path, patient_file_info=None):
    if patient_file_info is None:
        patient_file_info = {}

    # print(os.listdir(path))
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    for p in directories:
        # print(p)

        # Find the segmentation file
        seg_f_name = None
        flair_f_name = None
        for f_name in os.listdir(os.path.join(path, p)):
            if f_name.endswith("bias_norm-label.nrrd"):
                seg_f_name = f_name
            elif f_name.endswith("bias_norm.nrrd"):
                flair_f_name = f_name

        if not flair_f_name:
            print("SKIPPING -- NO FLAIR FOUND FOR:", p)
            continue
        
        print(p)
        print(seg_f_name)
        print(flair_f_name)
        print("\n")
        patient_file_info[p] = {"flair_file": flair_f_name, "seg_file": seg_f_name, "data_dir": path}
    return patient_file_info

def convert_to_nifti(input_file_full_path, output_file_full_path, seg):

    if seg == False:
        image = sitk.ReadImage(input_file_full_path,
                               # sitk.sitkInt16,
                               # "NrrdImageIO"
                               )
    else:
        image = sitk.ReadImage(input_file_full_path,
                               sitk.sitkInt16,
                               "NrrdImageIO"
                               )
    sitk.WriteImage(image, output_file_full_path, imageIO="NiftiImageIO")

def reorient_volume(input_file_full_path, output_file_full_path, orientation, slicer_dir):
    module_name = 'OrientScalarVolume'
    orientation_command = ['"' +slicer_dir+ '"', '--launch', module_name,  '"' + input_file_full_path + '" "' + output_file_full_path + '"', '-o', orientation]
    subprocess.run(' '.join(orientation_command),
                   shell=True,
                   stdout=subprocess.DEVNULL,
                   check=True)


def register_volume(input_file_full_path, output_file_full_path, transform_file_full_path, fixed_volume, transform_type, slicer_dir, interpolation_mode):
    affine_registration_command = ['"' + slicer_dir + '"','--launch', 'BRAINSFit', '--fixedVolume', '"' + fixed_volume + '"', '--movingVolume', '"' + input_file_full_path + '"', '--transformType', transform_type, '--initializeTransformMode useCenterOfHeadAlign', '--interpolationMode', interpolation_mode , '--outputVolume', '"' + output_file_full_path + '"', '--outputTransform', '"'+transform_file_full_path+'"']
    subprocess.run(' '.join(affine_registration_command), shell=True, stdout=subprocess.DEVNULL, check=True)


# Works for both segmentation and image itself
def resample_volume(input_file_path, output_file_full_path, reference_path,  interpolation_mode, slicer_dir):
    module_name = 'ResampleScalarVectorDWIVolume'


    resample_scalar_volume_command = ['"' + slicer_dir + '"', '--launch', module_name,
                                      '"' + input_file_path + '" "' + output_file_full_path + '"', '-i',
                                      interpolation_mode, '-R', '"' + reference_path + '"']
    subprocess.run(' '.join(resample_scalar_volume_command), shell=True, stdout=subprocess.DEVNULL, check=True)

def register_segmentation(input_file_path, output_file_full_path, reference_path, transform, interpolation_mode, slicer_dir,
                          use_reference=True, use_transform=True):
    module_name = 'ResampleScalarVectorDWIVolume'

    if use_reference & use_transform:
        resample_scalar_volume_command = ['"' + slicer_dir + '"', '--launch', module_name,
                                          '"' + input_file_path + '" "' + output_file_full_path + '"', '-i',
                                          interpolation_mode, '-f', '"' + transform + '"', '-R', '"' + reference_path + '"']
    elif ((use_transform) & (not use_reference)):
        resample_scalar_volume_command = ['"' + slicer_dir + '"', '--launch', module_name,
                                          '"' + input_file_path + '" "' + output_file_full_path + '"', '-i',
                                          interpolation_mode, '-f', '"' + transform + '"']
    elif ((not use_transform) & (use_reference)):
        resample_scalar_volume_command = ['"' + slicer_dir + '"', '--launch', module_name,
                                          '"' + input_file_path + '" "' + output_file_full_path + '"', '-i',
                                          interpolation_mode, '-R', '"' + reference_path + '"']
    else:
        sys.exit("UNSUPPORTED OPTION FOR REGISTERING THE SEGMENTATION")

    subprocess.run(' '.join(resample_scalar_volume_command), shell=True, stdout=subprocess.DEVNULL, check=True)


def skull_stripping(input_file_full_path,bet_path):
    # Need to install HD-BET package: https://github.com/MIC-DKFZ/HD-BET
    # git clone https://github.com/MIC-DKFZ/HD-BET then cd HD-BET and pip install -e .

    output_file_full_path = input_file_full_path.replace(".nii.gz", "_skull.nii.gz")
    # skull_stp_command = ['python','"' +  bet_path + '"', '-i', '"' + input_file_full_path + '"', '-o', '"' + output_file_full_path + '"']#, '-bet', '1']#, '-device cpu -mode fast -tta 0']
    skull_stp_command = ['python','"' +  bet_path + '"', '-i', '"' + input_file_full_path + '"', '-o', '"' + output_file_full_path + '"', '-device', 'cpu', '--disable_tta', '--save_bet_mask']#, '-bet', '1']#, '-device cpu -mode fast -tta 0']
    # print(skull_stp_command)
    # print(' '.join(skull_stp_command))
    subprocess.run(' '.join(skull_stp_command),
                   shell=True,
                   stdout=subprocess.DEVNULL,
                   check=True)

    #find directory name to find and return skull stripped image and brain mask
    dir_name = os.path.dirname(input_file_full_path)

    for file in os.listdir(dir_name):
        if file.endswith("skull.nii.gz"):
            skull = os.path.join(dir_name, file)
        elif file.endswith("skull_bet.nii.gz"):
            mask = os.path.join(dir_name, file)

    return skull, mask


# Function to perform N4 bias correction
# can pass skull stripped mask as mask_image 
def n4_bias_correction(input_file_path, output_file_full_path, mask_image=None):
    
    if mask_image != None:
        bias_correction_command = ['"' + slicer_dir + '"','--launch', 'N4ITKBiasFieldCorrection', '"' + input_file_path + '"', '"' + output_file_full_path + '"', '--maskimage', '"' + mask_image + '"']
    else:
        bias_correction_command = ['"' + slicer_dir + '"','--launch', 'N4ITKBiasFieldCorrection', '"' + input_file_path + '"', '"' + output_file_full_path + '"']

    subprocess.run(' '.join(bias_correction_command), shell=True, stdout=subprocess.DEVNULL, check=True)


def determine_output_file(output_dir, input_file_full_path, operation):
    input_file_name = os.path.basename(input_file_full_path)
    if ".nrrd" in input_file_name:
        output_file_name = input_file_name[:input_file_name.find('.nrrd')] + "_" + operation + ".nii.gz"
    elif ".nii.gz" in input_file_name:
        output_file_name = input_file_name[:input_file_name.find('.nii.gz')]+ "_" + operation  + ".nii.gz"
    else:
        sys.exit("Unidentified file type")

    output_file_full_path = os.path.join(output_dir, output_file_name)

    return output_file_full_path

if __name__ == '__main__':
    # Load in file names for FLAIR images and ROIs
    flair_and_seg_info = find_flair_and_seg_files(data_dir1)
    # flair_and_seg_info = find_flair_and_seg_files(data_dir2, flair_and_seg_info)
    print(flair_and_seg_info)

    # Create directory for writing data
    try:
        os.mkdir(output_path)
    except:
        print("Output Directory already exists")
    print(flair_and_seg_info)

    # Loop through all patients
    for patient_id in sorted(list(flair_and_seg_info.keys())):
        try:
            bias_corrected_flair_full_path = None
            registered_flair_full_path = None
            # Preprocess FLAIR image

            print("\nPatient: " + patient_id)
            print(flair_and_seg_info[patient_id])
            output_path_patient = os.path.join(output_path, patient_id)
            # Make directory for this patient
            try:
                os.mkdir(output_path_patient)
                os.mkdir(os.path.join(output_path_patient,"FLAIR"))
            except:
                print("Directory already exists for patient " + str(patient_id))
                if skip_if_already_done:
                    continue

            raw_flair_full_path = os.path.join(flair_and_seg_info[patient_id]["data_dir"], patient_id, flair_and_seg_info[patient_id]["flair_file"])
            raw_seg_full_path = os.path.join(flair_and_seg_info[patient_id]["data_dir"], patient_id, flair_and_seg_info[patient_id]["seg_file"])

            # Convert to FLAIR to nifti
            image_to_nifti_full_path = raw_flair_full_path # In case you want change the order
            original_img_full_path = determine_output_file(os.path.join(output_path_patient,"FLAIR"), image_to_nifti_full_path, "original")
            convert_to_nifti(image_to_nifti_full_path, original_img_full_path, False)
            print("CONVERTED THE IMAGE")

            # Convert Segmentation to nifti
            seg_to_nifti_full_path = raw_seg_full_path
            original_seg_full_path = determine_output_file(os.path.join(output_path_patient,"FLAIR"), seg_to_nifti_full_path, "original")
            convert_to_nifti(seg_to_nifti_full_path, original_seg_full_path, True)
            print("CONVERTED THE SEGMENTATION")



            # Reorient FLAIR sequence
            image_to_reorient_full_path = original_img_full_path # In case you want change the order
            reoriented_img_full_path = determine_output_file(os.path.join(output_path_patient,"FLAIR"), image_to_reorient_full_path, "reoriented")
            reorient_volume(image_to_reorient_full_path,reoriented_img_full_path,img_orientation, slicer_dir)
            print("REORIENTED THE IMAGE")

            # Reorient the segmentation
            seg_to_reorient_full_path = original_seg_full_path
            reoriented_seg_full_path = determine_output_file(os.path.join(output_path_patient,"FLAIR"), seg_to_reorient_full_path, "reoriented")
            reorient_volume(seg_to_reorient_full_path, reoriented_seg_full_path, img_orientation, slicer_dir)
            print("REORIENTED THE SEGMENTATION")

            # # Resample the FLAIR sequence
            # image_to_resample_full_path = reoriented_img_full_path
            # resampled_image_full_path = determine_output_file(os.path.join(output_path_patient, "FLAIR"), image_to_resample_full_path, "resampled")
            # resample_volume(image_to_resample_full_path, resampled_image_full_path, atlas, interpolation_mode_resample, slicer_dir)
            # print("RESAMPLED THE IMAGE")
            #
            # # Resample the segmentation
            # seg_to_resample_full_path = reoriented_seg_full_path
            # resampled_seg_full_path = determine_output_file(os.path.join(output_path_patient, "FLAIR"), seg_to_resample_full_path, "resampled")
            # resample_volume(seg_to_resample_full_path, resampled_seg_full_path, atlas, interpolation_mode_resample, slicer_dir)
            # print("RESAMPLED THE SEGMENTATION")

            # Register FLAIR sequence
            image_to_register_full_path = reoriented_img_full_path
            registered_flair_full_path = determine_output_file(os.path.join(output_path_patient,"FLAIR"), image_to_register_full_path, "registered")
            transformation_full_path = os.path.join(output_path_patient,"FLAIR","registration_transformation.h5")
            register_volume(image_to_register_full_path, registered_flair_full_path, transformation_full_path,
                            atlas, transform_type, slicer_dir, interpolation_mode_register_image)
            print("REGISTERED THE IMAGE")


            # Registering the segmentation using the transformation from the registered FLAIR image
            seg_to_register_full_path = reoriented_seg_full_path
            registered_seg_full_path = determine_output_file(os.path.join(output_path_patient, "FLAIR"), seg_to_register_full_path, "registered")
            register_segmentation(seg_to_register_full_path,
                                  registered_seg_full_path,
                                  registered_flair_full_path,
                                  transformation_full_path,
                                  interpolation_mode_register_segmentation,
                                  slicer_dir)
            print("REGISTERED THE SEGMENTATION")

            # Bias correction
            image_to_bias_correct_full_path = registered_flair_full_path
            skull, brain_mask = skull_stripping(image_to_bias_correct_full_path, bet_path) #Bias correction performs better with brain mask
            bias_corrected_flair_full_path = determine_output_file(os.path.join(output_path_patient, "FLAIR"), image_to_bias_correct_full_path, "bias_corrected")
            n4_bias_correction(image_to_bias_correct_full_path,
                               bias_corrected_flair_full_path,
                               mask_image = brain_mask)
            os.remove(skull)
            os.remove(brain_mask)
            print("BIAS CORRECTED THE IMAGE")



            # Skull stripping
            skull_stripped_flair_full_path, brain_mask_f = skull_stripping(bias_corrected_flair_full_path,bet_path)
            print("SKULL STRIPPED THE IMAGE")

            preprocessed_image = nib.load(skull_stripped_flair_full_path).get_fdata()
            preprocessed_image_mean = preprocessed_image.ravel().mean()
            preprocessed_image_stdev = preprocessed_image.ravel().std()

            # normalized_image = exposure.equalize_hist(
            #     (preprocessed_image - preprocessed_image.mean()) / preprocessed_image.std())

            # equal_hist_image = exposure.equalize_hist(preprocessed_image)

            # normalized_image = (equal_hist_image - equal_hist_image.mean()) / equal_hist_image.std()
            normalized_image = (preprocessed_image - preprocessed_image.mean()) / preprocessed_image.std()

            print("MAX: " + str(round(preprocessed_image.ravel().max(), 2)))
            print("MIN: " + str(round(preprocessed_image.ravel().min(), 2)))
            print("MEAN: " + str(round(preprocessed_image_mean, 2)))
            print("STDEV: " + str(round(preprocessed_image_stdev, 2)))

            with open(str(os.path.join(os.path.dirname(bias_corrected_flair_full_path),'preprocessed_FLAIR.npy')), 'wb') as f:
                np.save(f,normalized_image)

            with open(str(os.path.join(os.path.dirname(bias_corrected_flair_full_path),'preprocessed_segmentation.npy')), 'wb') as f:
                preprocessed_segmentation = nib.load(registered_seg_full_path).get_fdata()
                np.save(f,preprocessed_segmentation)


            os.remove(reoriented_seg_full_path)
            os.remove(reoriented_img_full_path)
            # os.remove(resampled_seg_full_path)
            # os.remove(resampled_image_full_path)
            os.remove(bias_corrected_flair_full_path)
            os.remove(brain_mask_f)
            os.remove(registered_flair_full_path)


        except Exception as e:
            print(e)
            exit()

