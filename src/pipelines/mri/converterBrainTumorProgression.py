import glob
import os

import cv2
import numpy as np
import pydicom.pixel_data_handlers.util as ddh
import yaml
from pydicom import dcmread
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



def main_Brain_Tumor_Progression(source_dir, result_dir):

    target_path = os.path.join(result_dir, "Brain_Tumor_Progression")
    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(target_path)
    # path to directory with images
    path = source_dir
    # Path to directory with images
    # # Extension of images
    extension = '.dcm'
    # # Searching for images recursively
    imgs = glob.glob(f"{path}/**/*{extension}", recursive=True)
    print(imgs)


    # DICOM conversion
    if extension == '.dcm':
        # iterate over found images
        for image in imgs:
            ds = dcmread(image)
            desc = ds.SeriesDescription
            print(desc)
            patID = ds.PatientID
            date = ds.StudyDate

            # skipping files, where mask can't be applied
            if (desc == 'T1post') or (desc == 'Mask_Tumor'):

                removeAtEnd = False
                # if image is not little endian implicit convert using gdcmconv

                if (ds.is_little_endian == False):
                    # converting using gdcm conv save as nameOfFile.converted
                    os.system(f'gdcmconv -w -X -I -i {image} -o {image}.converted;')
                    ds = dcmread(f'{image}.converted')
                    # set property to remove *.converted file at the end
                    removeAtEnd = True

                # if window parameters are provided use it to remove redundant data
                array = ds.pixel_array
                # apply modality LUT
                windowed = ddh.apply_modality_lut(array, ds)

                WindowCenter = ds.WindowCenter
                WindowWidth = ds.WindowWidth

                # apply window
                windowed = np.clip(windowed, WindowCenter - WindowWidth / 2, WindowCenter + WindowWidth / 2)
                # convert from hounsfield scale (-1000 to 1000) to png scale (0 to 255)
                min = np.min(windowed)
                if min < -1000: min = -1000
                windowed = windowed - min
                maks = np.max(windowed)
                ratio = maks / 255
                windowed = np.divide(windowed, ratio).astype(int)
                # save image as nameOfFile.png
                if not os.path.exists(os.path.join(target_path,  f'{desc}')):
                    os.makedirs(os.path.join(target_path,  f'{desc}'))
                cv2.imwrite(os.path.join(target_path,  f'{desc}/10{patID}{date}{image[-6:-4]}_{desc}.png'), windowed)

                # if created .converted file remove it
                if (removeAtEnd):
                    os.system(f'rm {image}.converted')
