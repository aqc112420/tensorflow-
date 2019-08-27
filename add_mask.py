'''
在原图上加上掩膜，便于直观化的展示
'''
import numpy as np
import cv2
import glob
from scipy.misc import imresize, imread
import matplotlib.pyplot as plt


def mask_overlay(image, mask, color=(0, 255, 0)):
    """
    Helper function to visualize mask on the top of the car
    """
    mask = mask * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img

if __name__ == '__main__':

    img_paths = glob.glob(r'E:\pycharmProjects\tensorflows\DIY\U-Net/*.jpg')
    for i in img_paths:
        img = imresize(imread(i, (1024, 1920), interp='nearest'))
        mask = imread(i.replace('0_images', '0_preds'))[:, :, :3]
        plt.imsave(i.replace('0_images', '0_images_mask'), mask_overlay(img, (mask > 0.5).astype(np.uint8)))
