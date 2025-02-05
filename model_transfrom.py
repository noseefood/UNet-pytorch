import torch

import argparse
import logging
import logging.config
from utils.load_conf import ConfigLoader
from pathlib import Path

from skimage import io, transform, color
import numpy as np
import os
import torch.nn.functional as F
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
from predict import predict_img


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

# load trained model
net = UNet(n_channels=3, n_classes=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device=device)
net.load_state_dict(torch.load('/home/xuesong/CAMP/US_servoing/no_ros/models/tracking_model/CP_epoch5.pth', map_location=device))

# trace model
img = cv2.imread('/home/xuesong/CAMP/segment/UNet-pytorch/data/test/pork15141.png')


mask = predict_img(net=net,
                    full_img=img,
                    scale_factor=0.5,
                    out_threshold=0.5,
                    device=device)

# mask = mask_to_image(full_mask)
plot_img_and_mask(img, np.uint8(mask), 0.3)

# print(out_img.shape,out_img)


# trace_module.save("data/models/traced_unet_model.pt")





