import os
import glob
import nibabel as nib
import cv2
import numpy as np
from pathlib import Path

import yaml
import os
from preprocessing.get_file_paths import GetFilePaths
from preprocessing.add_new_ids import AddNewIds
from preprocessing.convert_nii2dcm import ConvertNii2Dcm
from preprocessing.convert_dcm2png import ConvertDcm2Png
from preprocessing.create_masks_from_xml import CreateMasksFromXML
from preprocessing.create_file_tree import CreateFileTree
from preprocessing.create_blank_masks import CreateBlankMasks
from preprocessing.delete_imgs_without_masks import DeleteImgsWithoutMasks
from sklearn.pipeline import Pipeline


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

def main_CT_ORG(source_dir, result_dir):

    target_path = os.path.join(result_dir, "CT-ORG")
    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(target_path)

    extension = '.gz'

    # Path to directory with images
    path = source_dir

    # Searching for images recursively
    files = glob.glob(f"{path}/**/*{extension}", recursive=True)

    dictlabels = {150: 'Liver', 160: 'Bladder', 170: 'Lungs', 127: 'Kidneys', 140: 'Bone', 50: 'Brain', 0: 'Nothing'}

    numbersslice = []
    partofbody = []

    # iterate over found nii files
    for file in files:

        pat0 = Path(file).resolve().parent
        pat1 = pat0.parent
        categoryname = str(pat0)[len(str(pat1)) + 1:]

        filename = file[len(str(pat0)) + 1:]

        # Iterating over masks And creating dictionary of labels based on most appearing part of body
        if categoryname == 'Masks':
            nii_img = nib.load(file)
            nii_data = nii_img.get_fdata()

            slices = nii_data.shape[2]
            for slc in range(slices):
                # getting number of nifti file
                number = filename.split('-')
                number = number[1].split('.')
                number = number[0]

                image = np.array(nii_data[:, :, slc], dtype=int)
                # Changing given mask levels to ones specified in google doc
                np.place(image, image == 1, 150)
                np.place(image, image == 2, 160)
                np.place(image, image == 3, 170)
                np.place(image, image == 4, 127)
                np.place(image, image == 5, 140)
                np.place(image, image == 6, 50)

                # Checking ocurance of certain body parts
                lbls, counts = np.unique(image, return_counts=True)

                if(len(lbls) != 1):
                    lbls = lbls[1:]
                    counts = counts[1:]

                label = dictlabels[lbls[np.argmax(counts)]]

                # # save image as nameOfFile.png
                if label != 'Nothing':
                    if not os.path.exists(target_path+f'{categoryname}'):
                        os.makedirs(target_path+f'{categoryname}')
                    cv2.imwrite(target_path+f'{categoryname}/13{number}s{slc}_{label}.png', image)

                numbersslice.append(f'{number}s{slc}')
                partofbody.append(label)

                print('Zapisano maskę')

    # creating dictionary of labels
    labels = {numbersslice[i]: partofbody[i] for i in range(len(numbersslice))}
    # creating dictionary of windows for certain body parts
    windowcenters = {'Liver': 30, 'Bladder': 50, 'Lungs': -550, 'Kidneys': 50, 'Bone': 1800, 'Brain': 40}
    windowwidths = {'Liver': 150, 'Bladder': 400, 'Lungs': 2000, 'Kidneys': 400, 'Bone': 400, 'Brain': 80}
    # iterate over found nii image files
    for file in files:

        pat0 = Path(file).resolve().parent
        pat1 = pat0.parent
        categoryname = str(pat0)[len(str(pat1)) + 1:]

        filename = file[len(str(pat0)) + 1:]

        if categoryname == 'imgs':
            nii_img = nib.load(file)
            nii_data = nii_img.get_fdata()

            slices = nii_data.shape[2]
            for slc in range(slices):
                number = filename.split('-')
                number = number[1].split('.')
                number = number[0]

                image = np.array(nii_data[:, :, slc], dtype=int)
                # getting label from dictionary
                label = labels[f'{number}s{slc}']

                if label != 'Nothing':

                    WindowCenter = windowcenters[label]
                    WindowWidth = windowwidths[label]
                    # apply window
                    windowed = np.clip(image, WindowCenter - WindowWidth / 2, WindowCenter + WindowWidth / 2)

                    min = np.min(windowed)
                    windowed = windowed - min
                    maks = np.max(windowed)
                    ratio = maks / 255
                    windowed = np.divide(windowed, ratio).astype(int)


                    # # save image as nameOfFile.png
                    if not os.path.exists(os.path.join(target_path, categoryname)):
                        os.makedirs(os.path.join(target_path, categoryname))
                    cv2.imwrite(os.path.join(target_path, categoryname)+'/13'+str(number)+str(slc)+str(label) +'.png', windowed)

                    print('Zapisano obrazek')
