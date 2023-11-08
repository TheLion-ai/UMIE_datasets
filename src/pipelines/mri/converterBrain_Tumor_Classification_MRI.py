import glob
import os
from pathlib import Path

import cv2
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



def main_Brain_Tumor_Classification(source_dir, result_dir):

    target_path = os.path.join(result_dir, "Brain_Tumor_Classification")
    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(target_path)
    # path to directory with images
    path = source_dir
    # Extension of images
    extension = '.jpg'
    # Searching for images recursively
    imgs = glob.glob(f"{path}/**/*{extension}", recursive=True)
    print(imgs)

    glioma_tumor = 0
    meningioma_tumor = 0
    no_tumor = 0
    pituitary_tumor = 0

    for img in imgs:
        pat0 = Path(img).resolve().parent
        pat1 = pat0.parent
        # name of tumor
        categoryname = str(pat0)[len(str(pat1))+1:]
        print(categoryname)
        image = cv2.imread(img)

        # creating id's for images with the same label
        if categoryname == 'pituitary_tumor':
            counter = pituitary_tumor
            pituitary_tumor += 1

        if categoryname == 'no_tumor':
            counter = no_tumor
            no_tumor += 1

        if categoryname == 'meningioma_tumor':
            counter = meningioma_tumor
            meningioma_tumor += 1

        if categoryname == 'glioma_tumor':
            counter = glioma_tumor
            glioma_tumor += 1

        if not os.path.exists(os.path.join(target_path, f'{categoryname}')):
            os.makedirs(os.path.join(target_path, f'{categoryname}'))
        cv2.imwrite(os.path.join(target_path, f'{categoryname}') + f'/09{counter}_{categoryname}.png', image)
