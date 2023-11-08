# changing names of files to standarise them
import os

import PIL
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



if __name__ == "__main__":
    dir_1 = ''  # images source directory
    dir_2 = ''  # masks source directory
    dir_d1 = '' # destination masks directory
    dir_d2 = '' # destination images directory
    dataset_id = '10'   # dataset identifier

    for filename in os.listdir(dir_1):
        filename = filename.replace('._', '')
        print(filename)
        # get mask name from file name
        m_name = filename.replace('T1post', '')
        m_name = m_name.replace('.png', '')
        m_name = m_name + 'Mask_Tumor.png'
        print(m_name)
        # first part of new name
        n_name = filename.replace('.png', '')
        n_name = n_name.replace('T1post', '')
        n_name = n_name.replace('_', '')
        n_name = n_name.replace('-', '')
        n_name = dataset_id + n_name
        # open file
        try:
            image = PIL.Image.open(dir_1 + filename)
            try:
                mask = PIL.Image.open(dir_2 + m_name)
                # check if mask is empty (non empty mask means tumor)
                if not mask.getbbox():
                    n_name = n_name + '_Brain.png'
                else:
                    n_name = n_name + '_Brain-Tumor.png'
                mask.save((dir_d1 + n_name), format="png")
                image.save((dir_d2 + n_name), format="png")
            except IOError:
                print('Not found mask')

        except IOError:
            print('Not found image')
