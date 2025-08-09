import os
import subprocess
import sys
import urllib.request
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("WARNING: CUDA not available. GroundingDINO will run very slowly.")


def load_SAM():
    cur_dir = os.getcwd()

    SAM_DIR = os.path.join(cur_dir, "segment-anything-2")
    SAM_CHECKPOINT_PATH = os.path.join(cur_dir, "sam2_hiera_base_plus.pth")

    url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"

    # Clone repo if not already present
    if not os.path.exists(SAM_DIR):
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/bsushsiwba/sam2.git",
                "segment-anything-2",
            ]
        )

    # Add repo to Python path so imports work
    sys.path.append(SAM_DIR)

    # Download checkpoint if missing
    if not os.path.isfile(SAM_CHECKPOINT_PATH):
        urllib.request.urlretrieve(url, SAM_CHECKPOINT_PATH)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = SAM_CHECKPOINT_PATH
    model_cfg = "sam2_hiera_b+.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    return predictor
