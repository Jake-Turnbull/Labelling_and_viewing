from pathlib import Path

import matplotlib.pyplot as plt
import pytest


import Image_label_and_view


root_dir = Path("/Users/jacob.turnbull/VS_code_projects/UKB_Eddy/Labelling_and_viewing/synthetic_dwi_dataset")

# Run the labeling and viewing function on the synthetic dataset
Image_label_and_view.diffusion_image_viewer(root_dir, show=True)
    
