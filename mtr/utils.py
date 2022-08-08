# Built-in Imports
from typing import Literal
import pathlib
import os

# Third-party Imports
import numpy as np
import torch
import torch.autograd
import cv2

# Constants
IMG_SCALE  = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

NUM_CLASSES = {'nyu': 40, 'kitti': 6}
CWD = pathlib.Path(os.path.abspath(__file__)).parent

def match_size_img(img:np.ndarray) -> np.ndarray:

    # Reduce the size of the input image
    reduced_img = cv2.resize(img, (640, 480))

    return reduced_img

def prepare_img(img:np.ndarray) -> torch.Tensor:

    # First whiten the image
    whiten_img = (img*IMG_SCALE - IMG_MEAN)/IMG_STD 

    # Then convert to Torch tensor
    torch_img = torch.autograd.Variable(
        torch.from_numpy(whiten_img.transpose(2,0,1)[None]),
        requires_grad=False
    ).float()

    return torch_img

def clean_seg(
        seg:torch.Tensor, 
        img:np.ndarray, 
        dataset:Literal['nyu','kitti']
    ):
    num_classes = NUM_CLASSES[dataset]
    return cv2.resize(
        seg[0, :num_classes].cpu().data.numpy().transpose(1,2,0), 
        img.shape[:2][::-1],
        interpolation=cv2.INTER_CUBIC
    )

def clean_depth(depth:torch.Tensor, img:np.ndarray):
    return np.abs(cv2.resize(
        depth[0, 0].cpu().data.numpy(), 
        img.shape[:2][::-1],
        interpolation=cv2.INTER_CUBIC
    ))

def clean_norm(norm:torch.Tensor, img:np.ndarray):
    norm = cv2.resize(norm[0].cpu().data.numpy().transpose(1, 2, 0),
                       img.shape[:2][::-1],
                       interpolation=cv2.INTER_CUBIC)
    out_norm = norm / np.linalg.norm(norm, axis=2, keepdims=True)
    ## xzy->RGB ##
    out_norm[:, :, 0] = ((out_norm[:, :, 0] + 1.) / 2.) * 255.
    out_norm[:, :, 1] = ((out_norm[:, :, 1] + 1.) / 2.) * 255.
    out_norm[:, :, 2] = ((1. - out_norm[:, :, 2]) / 2.) * 255.
    out_norm = out_norm.astype(np.uint8)
    return out_norm

def get_cmap(dataset:Literal['nyu','kitti']):

    # Load the cmap for 'nyu'
    if dataset == 'nyu':
        cmap = np.load(CWD/'cmap_nyud.npy')
    elif dataset == 'kitti':
        cmap = np.load(CWD/'cmap_kitti.npy')
    else:
        raise Exception("Invalid dataset type")

    return cmap
