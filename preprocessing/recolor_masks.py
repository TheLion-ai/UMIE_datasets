"""Recolors masks from default color to the color specified in the config."""
import os
import glob

import cv2
import numpy as np
import yaml
import tqdm


def recolor_masks(
        source_path: str,
        target_path: str,
        mask: str,
        extension: str = '.png',
        source_color: int = 255
    ):
    """
    Recolors masks from default color to the color specified in the config.
    UMIE datasets consists of masks from several opensource datasets. Each type of masks has unique color encoding.
    To find if the mask has encoding, check the config file. If the mask is not in the config, add it.

    Args:
        source_path (str): path to the folder with masks
        target_path (str): path to the folder where recolored masks will be saved
        mask (str): name of the mask, check masks config for the list of available masks and add new ones if needed
        extension (str): extension of the mask files (only images supported)
        source_color (int): color of the mask to be changed
    """

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    masks_config = yaml.load(open('config/masks_config.yaml'), Loader=yaml.FullLoader)
    target_color = masks_config[mask]

    # Searching for images recursively
    files = glob.glob(f"{source_path}/**/*{extension}", recursive=True)


    for file in files:
        file_name = os.path.basename(file)
        new_path = os.path.join(target_path, file_name)
        if not os.path.exists(new_path):
            img = cv2.imread(file)
            # changing pixel values
            np.place(img, img == source_color, target_color)
            cv2.imwrite(new_path, img)
