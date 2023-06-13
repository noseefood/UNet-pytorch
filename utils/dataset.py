from os.path import splitext
from os import listdir
import os
import numpy as np
from glob import glob
import torch
import cv2
from torch.utils.data import Dataset
import logging

# import linecache  # 为了读取txt指定行

# TODO：写入数据增强的代码

_logger = logging.getLogger(__name__)

from PIL import Image

import random

from torchvision import transforms
import torchvision.transforms.functional as TF

import albumentations as A  # Augmentation


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix='_mask'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix

        # self.index_path = index_path # 

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)    # os.path.splitext(“文件路径”) 分离文件名与扩展名  会自动读取imgs下的所有文件的名称
                    if not file.startswith('.')]
        _logger.info(f'Creating dataset with {len(self.ids)} examples')

        self.transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.OneOf([
                        # 高斯噪点
                        A.IAAAdditiveGaussianNoise(),
                        A.GaussNoise(),], p=0.2),
                    
                ])

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, gray=False):

        # scale
        h, w, c = pil_img.shape
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newW, newH))
        #  scale image
        pil_img = cv2.resize(pil_img, (newW, newH))
        
        # convert to 1 channel
        if gray:
            pil_img = cv2.cvtColor(pil_img, cv2.COLOR_BGR2GRAY)

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        
        # 归一化(normalize) 0-1
        if img_trans.max() > 1:
            img_trans = img_trans / 255  

        return img_trans

    def __getitem__(self, i):
        # 根据index读取图片

        idx = self.ids[i] # 图片的名称
    
        mask_file = os.path.join(self.masks_dir, idx + '.png')   # *(path, filename)  我们使用了相同名字的imgs和masks文件
        img_file = os.path.join(self.imgs_dir, idx + '.png')

        # XXX: 这里mask_file 应该是路径名
        # assert len(mask_file) == 1, \
        #     f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        # assert len(img_file) == 1, \
        #     f'Either no image or multiple images found for the ID {idx}: {img_file}'
        # mask = Image.open(mask_file[0])
        # img = Image.open(img_file[0])
        mask = cv2.imread(mask_file)
        img = cv2.imread(img_file)

        assert img is not None, "img 为空，请检查图片是否在该路径下：%s" % img_file
        assert mask is not None, "mask 为空，请检查图片是否在该路径下：%s" % mask_file

        assert img.shape == mask.shape, \
            f'Image 和 mask {idx} 应该有相同的大小, 但这里img是: {img.shape}, mask是: {mask.shape}'
    
        
        #############albumentations#############
        transformed = self.transform(image=img, mask=mask)  # 必须一起输入才能保证对img/mask统一的变换
        img = transformed["image"]
        mask = transformed["mask"]
        #############albumentations#############

        # preprocess
        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale, gray=True)  # 这里可以强制图片转换为灰度图再输入网络，并且进行了归一化

        assert img.shape[1] == mask.shape[1], 'img and mask must have same height   %s' % idx
        assert img.shape[2] == mask.shape[2], 'img and mask must have same width    %s' % idx

        
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),  # torch.from_numpy()这种方法互相转的Tensor和numpy对象共享内存，修改一个，另一个也会改变
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }
        # return {
        #     'image': img_pil,  # torch.from_numpy()这种方法互相转的Tensor和numpy对象共享内存，修改一个，另一个也会改变
        #     'mask': mask_pil
        # }
    
    # def augment(self, image, flipCode):
    #     # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
    #     flip = cv2.flip(image, flipCode)
    #     return flip

    # def transform(self, image, mask):
    #     # Resize
    #     resize = transforms.Resize(size=(520, 520))
    #     image = resize(image)
    #     mask = resize(mask)

    #     # Random crop
    #     i, j, h, w = transforms.RandomCrop.get_params(
    #         image, output_size=(512, 512))
    #     image = TF.crop(image, i, j, h, w)
    #     mask = TF.crop(mask, i, j, h, w)

    #     # Random horizontal flipping
    #     if random.random() > 0.5:
    #         image = TF.hflip(image)
    #         mask = TF.hflip(mask)

    #     # Random vertical flipping
    #     if random.random() > 0.5:
    #         image = TF.vflip(image)
    #         mask = TF.vflip(mask)

    #     # Transform to tensor
    #     image = TF.to_tensor(image)
    #     mask = TF.to_tensor(mask)
    #     return image, mask
    

if __name__ == '__main__':
    data = BasicDataset('/data1/volume1/data/xuesong/Tracking/UNet-pytorch/data/imgs', '/data1/volume1/data/xuesong/Tracking/UNet-pytorch/data/masks')
    for n, i in enumerate(data):
        print(n)




