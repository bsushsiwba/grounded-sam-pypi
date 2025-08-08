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

    AUTODISTILL_CACHE_DIR = os.path.expanduser("~/.cache/autodistill")
    SAM_CACHE_DIR = os.path.join(AUTODISTILL_CACHE_DIR, "segment_anything_2")
    SAM_CHECKPOINT_PATH = os.path.join(SAM_CACHE_DIR, "sam2_hiera_base_plus.pth")

    url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"

    os.makedirs(SAM_CACHE_DIR, exist_ok=True)
    os.chdir(SAM_CACHE_DIR)

    repo_path = os.path.join(SAM_CACHE_DIR, "segment-anything-2")

    # Clone repo if it doesn't exist
    if not os.path.exists(repo_path):
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/bsushsiwba/sam2.git",
                "segment-anything-2",
            ]
        )

    # Append cloned repo to sys.path so imports work
    sys.path.append(repo_path)

    # Download checkpoint if it doesn't exist
    if not os.path.isfile(SAM_CHECKPOINT_PATH):
        urllib.request.urlretrieve(url, SAM_CHECKPOINT_PATH)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = SAM_CHECKPOINT_PATH
    model_cfg = "sam2_hiera_b+.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    os.chdir(cur_dir)
    return predictor
