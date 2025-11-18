from argparse import ArgumentParser
import os
from utils import nrrd_to_npy, dicom_to_npy, nifti_to_npy
from dataclasses import dataclass, field, fields
import warnings
import json
import numpy as np


@dataclass
class dataProcessConfig():
    # patient level paths, within the patient folder there is Segmentation.seg.nrrd (mask)
    # rename output files to segmentation/flair_patient_ID
    # not guaranteed to be Segmentation exactly, but will contain Segmentation in the title, or will have file extension .seg.nrrd
    # the flair sequence is 5 AX FLAIR.nrrd (not exactly, but should contain FLAIR)
    medullo_path: str = "Z:/Datasets/MedicalImages/BrainData/SickKids/Medulloblastoma/MRI"
    
    # patient_id > 8 AX FLAIR.nrrd is the flair sequence, 6_post_bias_nom-label.nrrd might be the mask? 
    dipg_path: str = "Z:/Datasets/MedicalImages/BrainData/SickKids/DIPG/segmentations"

    # patient_id > FLAIR > preprocessed_FLAIR.npy is the FLAIR, preprocessed_segmentation.npy is the segmentation
    plgg_path: str = "Z:/Datasets/MedicalImages/BrainData/SickKids/preprocessed_pLGG_EN_Nov2023_KK"
    output_path: str = "./test_output"
    save_to_jsons: bool = False

def save_dict_to_json(data_dict, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(data_dict, f, indent=4)

    print(f"Saved JSON to: {output_file}")

def collect_medullo_data(root_folder):
    patient_dict = {}

    # Traverse all folders
    for dirpath, _, filenames in os.walk(root_folder):
        patient = os.path.basename(dirpath)

        flair_path = None
        seg_path = None

        for f in filenames:
            lower = f.lower()
            full_path = os.path.join(dirpath, f)

            # FLAIR file
            if "flair" in lower and lower.endswith(".nrrd"):
                flair_path = full_path

            # Segmentation file
            if "segmentation" in lower and lower.endswith(".seg.nrrd"):
                seg_path = full_path

        # If this directory has anything relevant, record it
        if flair_path or seg_path:
            # Ensure patient entry exists
            if patient not in patient_dict:
                patient_dict[patient] = {"flair": None, "seg": None}

            # store found files
            if flair_path:
                patient_dict[patient]["flair"] = flair_path
            if seg_path:
                patient_dict[patient]["seg"] = seg_path

    # Now validate every patient
    validated_dict = {}
    for patient, files in patient_dict.items():
        if files["flair"] and files["seg"]:
            validated_dict[patient] = files
        else:
            warnings.warn(
                f"[WARNING] Missing data for Medullo patient '{patient}': "
                f"FLAIR found? {bool(files['flair'])} | SEG found? {bool(files['seg'])}. "
                "Skipping patient."
            )

    return validated_dict

def collect_dipg_data(root_folder):
    patient_dict = {}

    # patient-level directories
    for patient in os.listdir(root_folder):
        patient_path = os.path.join(root_folder, patient)
        if not os.path.isdir(patient_path):
            continue

        patient_flair = None
        patient_seg = None

        # scan inside the patient directory
        for dirpath, _, filenames in os.walk(patient_path):
            for f in filenames:
                lower = f.lower()
                full_path = os.path.join(dirpath, f)

                # FLAIR file
                if "flair" in lower and lower.endswith(".nrrd"):
                    patient_flair = full_path

                # Segmentation file
                if "bias_norm-label" in lower and lower.endswith(".nrrd"):
                    patient_seg = full_path

        # Validate
        if patient_flair and patient_seg:
            patient_dict[patient] = {
                "flair": patient_flair,
                "seg": patient_seg
            }
        else:
            warnings.warn(
                f"[WARNING] Missing data for patient '{patient}': "
                f"FLAIR found? {bool(patient_flair)} | SEG found? {bool(patient_seg)}. "
                "Skipping patient."
            )

    return patient_dict

def collect_plgg_data(root_folder):
    patient_dict = {}

    for patient in os.listdir(root_folder):
        patient_path = os.path.join(root_folder, patient)
        if not os.path.isdir(patient_path):
            continue

        flair_folder = os.path.join(patient_path, "FLAIR")
        if not os.path.isdir(flair_folder):
            warnings.warn(
                f"[WARNING] Patient '{patient}' has no FLAIR folder. Skipping."
            )
            continue

        patient_flair = None
        patient_seg = None

        # Search inside the FLAIR subfolder
        for f in os.listdir(flair_folder):
            lower = f.lower()
            full_path = os.path.join(flair_folder, f)

            # FLAIR npy file
            if "preprocessed_flair" in lower and lower.endswith(".npy"):
                patient_flair = full_path

            # Segmentation npy file
            if "preprocessed_segmentation" in lower and lower.endswith(".npy"):
                patient_seg = full_path

        # Validate
        if patient_flair and patient_seg:
            patient_dict[patient] = {
                "flair": patient_flair,
                "seg": patient_seg
            }
        else:
            warnings.warn(
                f"[WARNING] Missing data for pLGG patient '{patient}': "
                f"FLAIR found? {bool(patient_flair)} | SEG found? {bool(patient_seg)}. "
                "Skipping this patient."
            )

    return patient_dict

if __name__ == "__main__":
    parser = ArgumentParser(dataProcessConfig)

    # auto-create CLI args from dataclass fields
    for field in fields(dataProcessConfig):
        parser.add_argument(
            f"--{field.name}",
            type=type(field.default),
            default=field.default,
        )
    
    args = parser.parse_args()
    cfg = dataProcessConfig(**vars(args))

    # get all input paths for plGG
    # plggg: should have 353 (maybe 397) --> got 468
    plgg_dict = collect_plgg_data(cfg.plgg_path)

    # get all input paths for DIPG
    # DIPG: should have 89 --> only got 47
    dipg_dict = collect_dipg_data(cfg.dipg_path)

    # get all input paths for medulloblastoma
    # medulloblastoma: should have 106 --> YEP
    medullo_dict = collect_medullo_data(cfg.medullo_path) 
    print("plgg samples:", len(plgg_dict), 
          "dipg samples:", len(dipg_dict), 
          "medullo samples:", len(medullo_dict))

    if cfg.save_to_jsons:
        save_dict_to_json(medullo_dict, os.path.join(cfg.output_path, "medullo_data.json"))
        save_dict_to_json(dipg_dict, os.path.join(cfg.output_path, "dipg_data.json"))
        save_dict_to_json(plgg_dict, os.path.join(cfg.output_path, "plgg_data.json"))

    # TODO: other dataset processing

    # run conversions + apply masks to get segmentated npys
    plgg_npys, dipg_npys, medullo_npys, = {}, {}, {} # plgg is already in npy

    # TODO: just for testing reasons, revert later
    max_items = 5
    i = 0

    for patient_id, nrrds in dipg_dict.items():
        i += 1
        flair = nrrd_to_npy(nrrds["flair"])
        mask = nrrd_to_npy(nrrds["seg"])
        print("dipg shape:", flair.shape)

        if not flair.shape == mask.shape:
            print("shape mismatch between flair and mask:", flair.shape, mask.shape)
        
        seg_flair = flair.copy()
        seg_flair[~mask] = 0
        dipg_npys[patient_id] = seg_flair

        if i >= max_items:
            i = 0
            break
    
    for patient_id, nrrds in medullo_dict.items():
        i += 1
        flair = nrrd_to_npy(nrrds["flair"])
        mask = nrrd_to_npy(nrrds["seg"])
        print("medullo shape:", flair.shape)
        if not flair.shape == mask.shape:
            print("shape mismatch between flair and mask:", flair.shape, mask.shape)

        seg_flair = flair.copy()
        seg_flair[~mask] = 0
        medullo_npys[patient_id] = seg_flair

        if i >= max_items:
            i = 0
            break

    # I think plgg already had the mask applied...
    for patient_id, npys in plgg_dict.items():
        i += 1
        seg = np.load(npys["seg"])
        plgg_npys[patient_id] = seg
        print("plgg shape:", seg.shape)

        if i >= max_items:
            i = 0
            break

    # resize to the same shape (if not same shape) --> crop to the segmentation area? but how...

    # divide to train/val/test

    # write to output data directory

    # NOTE: will probably need to save to the shared drive (try to save a smaller 
    # subet locally first)

    # Desired structure:
    # train 
    # -- medulloblastoma
    #      > masked_flair_patientID.npy
    # -- DIPG
    #      > masked_flair_patientID.npy
    # ... etc you get the point
    # test
    # ... same format  
    print("Finished processing pipeline.")