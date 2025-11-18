from argparse import ArgumentParser
import os
from data.utils import nrrd_to_npy, dicom_to_npy, nifti_to_npy
from dataclasses import dataclass, field, fields

@dataclass
class dataProcessConfig():
    input_path: str = ""
    output_path: str = ""

if __name__ == "main":
    parser = ArgumentParser(dataProcessConfig)
    config = parser.parse_args()

    # get all input paths for plGG

    # get all input paths for DPIG

    # get all input paths for medulloblastoma 

    # run conversions

    # resize to the same shape

    # divide to train/val/test

    # write to output data directory

    # NOTE: will probably need to save to the shared drive (try to save a smaller 
    # subet locally first)
    print("Finished processing pipeline.")