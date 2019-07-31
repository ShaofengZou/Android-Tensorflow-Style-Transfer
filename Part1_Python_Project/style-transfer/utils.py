#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: utils.py

import scipy.misc
import numpy as np
import os
import cv2

def get_files(img_dir):
    files = list_files(img_dir)
    return [os.path.join(img_dir, x) for x in files]


def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)


# def scale_img(style_path, style_scale):
#     scale = float(style_scale)
#     o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
#     scale = float(style_scale)
#     new_shape = (int(o0 * scale), int(o1 * scale), o2)
#     style_target = _get_img(style_path, img_size=new_shape)
#     return style_target


def get_img(src, img_size=False):
	img = cv2.imread(src)[:,:,(2,1,0)]
	if not (len(img.shape) == 3 and img.shape[2] == 3):
		img = np.dstack((img, img, img))
	if img_size is not False:
		img_target_size = (img_size[0],img_size[1])
		img = cv2.resize(img,img_target_size,interpolation = cv2.INTER_CUBIC)
	return img


def exists(p, msg):
    assert os.path.exists(p), msg


def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break
    return files
