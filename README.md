# Diffusion image viewer and labeller

This is a set of scripts for viewing and recording bad volumes in diffusion MRI data. AI has been used to convert and extend much of this code from the original MATLAB versions which are stored on the FMRIB compute cluster into Python. The AI used for this was ChatGPT (specifically, GPT 5 mini for conversion).

Future development may use Codex and Agentic AI.
Install by downloading directly or by running: pip install git+https://github.com/Jake-Turnbull/Labelling_and_viewing.git in the terminal.


This code was created by - **Jake Turnbull**: [jacob.turnbull@ndcn.ox.ac.uk](mailto:jacob.turnbull@ndcn.ox.ac.uk)

## FSLeyes Diffusion QC

The `fsleyes_diffusion_qc` package provides a ranked-volume review engine, crash-conscious JSON project storage, CSV exports, and a dockable FSLeyes control panel. The plugin API was checked against FSLeyes 1.20.1. Install the package into the same Python environment used to launch FSLeyes:

```bash
python -m pip install -e .
fsleyes --showAllPlugins
```

In FSLeyes, open the Diffusion QC control from the orthographic view's Settings menu. Select the subject CSV, volume-ranking CSV, subject root, and results directory, then choose Load / Resume. The project JSON and exports are written only to the results directory. The plugin reads `SubjectID/dMRI/raw/AP.nii.gz` (or `AP.nii`), `AP.bval`, and optionally `AP.bvec`.

Generate a small demo cohort with three artificial subjects and 105 volumes each:

```bash
diffusion-qc-generate-synthetic --output synthetic_dwi_dataset --subjects 3
```

This creates `synthetic_subjects.csv`, `synthetic_volume_ranking.csv`, and artificial `AP.nii.gz`, `AP.bval`, and `AP.bvec` files under each subject. The images are test phantoms, not realistic or clinical scans. Run the tests with `python -m pytest`.
