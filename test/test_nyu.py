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
import matplotlib.pyplot as plt

# Internal Imports
import mtr

# Constants
CWD = pathlib.Path(os.path.abspath(__file__)).parent
EXAMPLES_DIR = CWD.parent/'examples'
DEVICE = torch.device('cpu')
    
# Load example data
NYU_MIN_DEPTH = 0
NYU_MAX_DEPTH = 80
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
    gt_seg = np.array(Image.open(seg_p))
    gt_depth = np.array(Image.open(depth_p))
    gt_norm = np.array(Image.open(norm_p))

    # Prepare the image
    img = mtr.match_size_img(img)
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

    # Load the cmap for 'nyu'
    cmap = mtr.get_cmap('nyu')
    depth_coeff = 5000

    # Create figure
    plt.figure(figsize=(18, 12))
    plt.subplot(171)
    plt.imshow(img)
    plt.title('orig img')
    plt.axis('off')
    plt.subplot(172)
    plt.imshow(cmap[gt_seg + 1])
    plt.title('gt segm')
    plt.axis('off')
    plt.subplot(173)
    plt.imshow(cmap[c_seg.argmax(axis=2) + 1].astype(np.uint8))
    plt.title('pred segm')
    plt.axis('off')
    plt.subplot(174)
    plt.imshow(gt_depth / depth_coeff, cmap='plasma', vmin=NYU_MIN_DEPTH, vmax=NYU_MAX_DEPTH)
    plt.title('gt depth')
    plt.axis('off')
    plt.subplot(175)
    plt.imshow(c_depth, cmap='plasma', vmin=NYU_MIN_DEPTH, vmax=NYU_MAX_DEPTH)
    plt.title('pred depth')
    plt.axis('off')
    plt.subplot(176)
    plt.imshow(gt_norm)
    plt.title('gt norm')
    plt.axis('off')
    plt.subplot(177)
    plt.imshow(c_norm)
    plt.title('pred norm')
    plt.axis('off')
    plt.show()
