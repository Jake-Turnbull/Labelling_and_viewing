import os
import numpy as np
import nibabel as nib

"""Genererates small synthetic DWI dataset with 3 subjects, each having 30 volumes. 

The volumes have a brain-like structure with some variation and injected "bad" volumes every 10th volume. 


The dataset is saved in NIfTI format under the specified root directory.

This can be used for testing preprocessing pipelines and artifact detection algorithms."""
def make_brain_like_volume(nx, ny, nz):
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    brain = np.exp(-(X**2 + Y**2 + Z**2) * 4)

    wm = np.exp(-((X*1.2)**2 + (Y*1.2)**2 + (Z*1.2)**2) * 6)
    gm = np.exp(-(X**2 + Y**2 + Z**2) * 3)

    signal = 0.6 * gm + 0.4 * wm

    return signal


def add_volume_variation(base, v, nvol):
    scale = 1 + 0.2 * np.sin(2 * np.pi * v / nvol)
    noise = np.random.normal(0, 0.05, base.shape)

    # inject occasional "bad" volumes
    if v % 10 == 0:
        noise += np.random.normal(0, 0.3, base.shape)
        scale *= 0.5

    return base * scale + noise


def generate_subject(nx=64, ny=64, nz=40, nvol=30):
    base = make_brain_like_volume(nx, ny, nz)

    dwi = np.zeros((nx, ny, nz, nvol), dtype=np.float32)

    for v in range(nvol):
        dwi[..., v] = add_volume_variation(base, v, nvol)

    return dwi


def save_nifti(data, filepath):
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, filepath)


def create_dataset(root_dir, n_subjects=3):
    for i in range(n_subjects):
        subj_name = f"sub-{i+1:02d}"
        subj_path = os.path.join(root_dir, subj_name, "dMRI", "raw")
        os.makedirs(subj_path, exist_ok=True)

        data = generate_subject()
        out_file = os.path.join(subj_path, "AP.nii.gz")

        save_nifti(data, out_file)


if __name__ == "__main__":
    output_root = "synthetic_dwi_dataset"
    create_dataset(output_root, n_subjects=3)