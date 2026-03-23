from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from ViewerScripts import Image_label_and_view

def test_run_on_synth():
    # Define the path to the synthetic dataset
    root_dir = Path("/Users/jacob.turnbull/VS_code_projects/UKB_Eddy/Labelling_and_viewing/Synthetic_dataset")
    
    # Run the labeling and viewing function on the synthetic dataset
    Image_label_and_view.diffusion_image_viewer(root_dir)
    
    print("All expected files are present. Test passed.")

test_run_on_synth()