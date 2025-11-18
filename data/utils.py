import numpy as np
import os
import nrrd
import pydicom  
import glob
import nibabel as nib 

def nrrd_to_npy(nrrd_path: str, npy_path: str = None):

    data, header = nrrd.read(nrrd_path)
    return data 

    # if npy_path is None:
    #     npy_path = os.path.splitext(nrrd_path)[0] + ".npy"

    # np.save(npy_path, data)
    # return npy_path

def dicom_to_npy(dicom_dir: str, npy_path: str = None):

    dicom_files = sorted(glob.glob(os.path.join(dicom_dir, "*.dcm")))
    if len(dicom_files) == 0:
        raise ValueError("No DICOM files found in directory: " + dicom_dir)

    # Read slices and sort by InstanceNumber
    slices = []
    for f in dicom_files:
        ds = pydicom.dcmread(f)
        slices.append((ds.InstanceNumber, ds.pixel_array))

    slices.sort(key=lambda x: x[0])
    volume = np.stack([s[1] for s in slices], axis=0)

    # if npy_path is None:
    #     npy_path = os.path.join(dicom_dir.rstrip("/"), "dicom_volume.npy")

    return volume
    # np.save(npy_path, volume)
    # return npy_path

def nifti_to_npy(nii_path: str, npy_path: str = None): 

    img = nib.load(nii_path)
    data = img.get_fdata()  # float64; use get_fdata(dtype=np.float32) if needed
    return data

    # if npy_path is None:
    #     if nii_path.endswith(".nii.gz"):
    #         npy_path = nii_path.replace(".nii.gz", ".npy")
    #     else:
    #         npy_path = nii_path.replace(".nii", ".npy")

    # np.save(npy_path, data)
    # return npy_path