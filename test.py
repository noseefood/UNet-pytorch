

import linecache  # 为了读取txt指定行
import os
import cv2


index_path = '/home/xuesong/CAMP/segment/UNet-pytorch/data/index.txt'


idx = 900

masks_dir = '/home/xuesong/CAMP/segment/UNet-pytorch/data/masks'
imgs_dir = '/home/xuesong/CAMP/segment/UNet-pytorch/data/imgs'


# s = str.replace('\n', '')
mask_Name = linecache.getline(index_path, idx)
mask_Name = str.replace(mask_Name, '\n', '')
img_Name = mask_Name

mask_file = os.path.join(masks_dir, mask_Name + '.png')   # *(path, filename),注意getline会将最后的换行符号也读取进来
img_file = os.path.join(imgs_dir, img_Name + '.png')

mask = cv2.imread(mask_file)
img = cv2.imread(img_file)

assert img is not None, "img 为空，请检查图片是否在该路径下：%s" % img_file
assert mask is not None, "mask 为空，请检查图片是否在该路径下：%s" % mask_file

cv2.imshow('img', img)

mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
cv2.imshow('mask', mask)

cv2.waitKey(0)
