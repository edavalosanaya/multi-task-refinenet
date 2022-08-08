# Built-in Imports
import os
import pathlib
import time
import logging

logger = logging.getLogger(__name__)

# Third-party Imports
import pytest
import torch
from PIL import Image
import numpy as np

# Internal Imports
import mtr

# Constants
CWD = pathlib.Path(os.path.abspath(__file__)).parent
EXAMPLES_DIR = CWD.parent/'examples'
DEVICE = torch.device('cuda')
    
# Load example data
NYU_DATA_DIR = EXAMPLES_DIR/'data'/'ExpNYUD_three'
img_p = NYU_DATA_DIR/'000433.png'
seg_p = NYU_DATA_DIR/'segm_gt_000433.png'
depth_p = NYU_DATA_DIR/'depth_gt_000433.png'
norm_p = NYU_DATA_DIR/'norm_gt_000433.png'

assert all([path.exists() for path in [img_p, seg_p, depth_p, norm_p]])

@pytest.fixture
def nyu_net():
    net = mtr.TrainedNet(
        dataset='nyu', 
        tasks=['seg', 'normals'],
        device=DEVICE
    )
    net.eval()
    return net

def test_nyu_forward_propagation(nyu_net):

    # Load img
    img = np.array(Image.open(img_p))

    # Prepare the image
    prep_img = mtr.prepare_img(img)

    # Move the image to the device
    prep_img = prep_img.to(DEVICE)

    # Perform forward propagation
    tic = time.time()
    seg, depth, norm = nyu_net(prep_img)
    toc = time.time()
    logger.info(f"Inference time: {toc-tic}")

    # Clean outputs
    c_seg = mtr.clean_seg(seg, img, 'nyu')
    c_depth = mtr.clean_depth(depth, img)
    c_norm = mtr.clean_norm(norm, img) 
