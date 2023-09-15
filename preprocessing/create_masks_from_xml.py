"""Converts masks from xml files to png images with appropriate color encoding."""

import os
import glob
import re

import yaml
import cv2

import untangle
import glob
import numpy as np
import csv
from PIL import Image, ImageDraw
import os
import PIL.Image
import plistlib

def create_masks_from_xml(
        source_path: str,
        target_path: str,
        dataset: str,
):
    dataset_uid = yaml.load(open('config/dataset_uid_config.yaml'), Loader=yaml.FullLoader)[dataset]
    dataset_masks = yaml.load(open('config/dataset_masks_config.yaml'), Loader=yaml.FullLoader)[dataset]
    mask_encoding_config = yaml.load(open('config/masks_encoding_config.yaml'), Loader=yaml.FullLoader)
    target_colors = {mask: mask_encoding_config[mask] for mask in dataset_masks}

    extension = "xml"
    files = glob.glob(f"{source_path}/**/*{extension}", recursive=True)
    for file in files:
        with open(file, mode="rb") as xml_file:
            segmentations = plistlib.load(xml_file)['Images']
        

        patient_id = os.path.basename(file).split('.')[0]
        filename_prefix = f"{dataset_uid}{patient_id}"
        

        pattern = r"[-+]?\d*\.\d+|\d+"
        for segmentation in segmentations:
            img_id = segmentation['ImageIndex']
            img = np.zeros((512, 512), np.uint8)
            for roi in segmentation['ROIs']:
                if roi['NumberOfPoints'] > 0:
                    points = []
                    for point in roi['Point_px']:
                        x, y = re.findall(pattern, point)
                        x, y = float(x), float(y)
                        x, y = int(x), int(y)
                        points.append([x, y])
                    cv2.fillPoly(img, [np.array(points)], (255))

            new_path = os.path.join(target_path, f"{filename_prefix}{str(img_id).zfill(4)}.png")
            cv2.imwrite(new_path, img)

def add_blank_masks():
    extension = "png"
    imgs = glob.glob(f"{os.path.join(source_path, 'Images')}/**/*{extension}", recursive=True)
    imgs = [os.path.basename(img) for img in imgs]
    masks = glob.glob(f"{os.path.join(source_path, 'Masks')}/**/*{extension}", recursive=True)
    masks = [os.path.basename(mask) for mask in masks]
    for img in imgs:
        if img not in masks:
            new_path = os.path.join(source_path, 'Masks', img)
            img = np.zeros((512, 512), np.uint8)
            cv2.imwrite(new_path, img)
