import glob
import os
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import yaml
from sklearn.pipeline import Pipeline

from preprocessing.add_new_ids import AddNewIds
from preprocessing.convert_dcm2png import ConvertDcm2Png
from preprocessing.convert_nii2dcm import ConvertNii2Dcm
from preprocessing.create_blank_masks import CreateBlankMasks
from preprocessing.create_file_tree import CreateFileTree
from preprocessing.create_masks_from_xml import CreateMasksFromXML
from preprocessing.delete_imgs_without_masks import DeleteImgsWithoutMasks
from preprocessing.get_file_paths import GetFilePaths


def preprocess_[DATASET_NAME](source_path, target_path):
    dataset_name = ""
    dataset_uid = yaml.load(open('config/dataset_uid_config.yaml'), Loader=yaml.FullLoader)[dataset_name]
    phases = yaml.load(open('config/phases_config.yaml'), Loader=yaml.FullLoader)[dataset_name]
    mask_encoding_config = yaml.load(open('config/masks_encoding_config.yaml'), Loader=yaml.FullLoader)
    dataset_masks = yaml.load(open('config/dataset_masks_config.yaml'), Loader=yaml.FullLoader)[dataset_name]

    mask_colors_old2new = {v: mask_encoding_config[k] for k, v in dataset_masks.items()}

    params = {
        "target_path": target_path,
        "masks_path": masks_path,
        "dataset_name": dataset_name,
        "dataset_uid": dataset_uid,
        "phases": phases,
        "dataset_masks": dataset_masks,
        "z-fill": 3,
        "img_id_extractor": lambda x: os.path.basename(x).split("-")[-1],
        "study_id_extractor": lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))),
        "mask_colors_old2new": mask_colors_old2new
    }

    pipeline = Pipeline(steps=[
        ("create_file_tree", CreateFileTree(**params)),
        ("convert_nii2dcm", ConvertNii2Dcm(**params)), # TODO: add conversion from nii to dcm
        ("get_file_paths", GetFilePaths(**params)),
        ("add_new_ids", AddNewIds(**params)),
        ("convert_dcm2png", ConvertDcm2Png(**params)),
        # Choose either to create blank masks or delete images without masks
        # ("create_blank_masks", CreateBlankMasks(**params)),
        ("delete_imgs_without_masks", DeleteImgsWithoutMasks(**params))
    ], )
    pipeline.transform(X=source_path)

def main_MosMed(source_dir, result_dir):

    target_path = os.path.join(result_dir, "MosMed")
    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(target_path)

    extension = '.gz'

    # Path to directory with images
    path = source_dir

    # Searching for images recursively
    files = glob.glob(f"{path}/**/*{extension}", recursive=True)

    # iterate over found nii files
    for file in files:
        nii_img = nib.load(file)
        nii_data = nii_img.get_fdata()

        slices = nii_data.shape[2]

        pat0 = Path(file).resolve().parent
        pat1 = pat0.parent
        categoryname = str(pat0)[len(str(pat1)) + 1:]

        filename = file[len(str(pat0))+1:]

        for slc in range(slices):
            image = np.array(nii_data[:, :, slc])
            pat0 = Path(file).resolve().parent
            pat1 = pat0.parent
            # name of directory containing nii file
            categoryname = str(pat0)[len(str(pat1)) + 1:]

            WindowCenter = -550
            WindowWidth = 2000
            # apply window
            windowed = np.clip(image, WindowCenter - WindowWidth / 2, WindowCenter + WindowWidth / 2)

            # convert from hounsfield scale (-1000 to 1000) to png scale (0 to 255)
            min = np.min(windowed)
            windowed = windowed - min
            maks = np.max(windowed)
            ratio = maks / 255
            windowed = np.divide(windowed, ratio).astype(int)

            if categoryname == 'masks':
                category = 'Masks'
                label = 'Parenchyma0025'
            # Choosing category name based on folder it's contained in
            else:
                category = 'Images'
                if categoryname == 'CT-0':
                    label = 'Good'
                elif categoryname == 'CT-1':
                    label = 'Parenchyma0025'
                elif categoryname == 'CT-2':
                    label = 'Parenchyma2550'
                elif categoryname == 'CT-3':
                    label = 'Parenchyma5075'
                elif categoryname == 'CT-4':
                    label = 'Parenchyma7500'


            # save image as nameOfFile.png
            if not os.path.exists(os.path.join(target_path, categoryname)):
                os.makedirs(os.path.join(target_path, categoryname))
            cv2.imwrite(os.path.join(target_path, categoryname)+f'/01{filename[6:10]}{slc}_Lungs-{label}.png', windowed)
