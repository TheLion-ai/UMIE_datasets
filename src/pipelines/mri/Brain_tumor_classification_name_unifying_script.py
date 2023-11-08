import glob
import os

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


if __name__ == '__main__':
    glioma = []
    meningioma = []
    good = []
    pituitary = []
    #put the directory here
    dir_to_new_db = r''
    #searching for pngs in our database
    for item in glob.glob("db\\glioma_tumor\\*.png"):
        glioma.append(item)
    for item in glob.glob("db\\meningioma_tumor\\*.png"):
        meningioma.append(item)
    for item in glob.glob("db\\no_tumor\\*.png"):
        good.append(item)
    for item in glob.glob("db\\pituitary_tumor\\*.png"):
        pituitary.append(item)

    #here we iterate over files to unify the names
    #and to put them in new directory
    for i,item in enumerate(glioma):
        words_old = item.split('\\')
        words_new = words_old.copy()
        iter  = str(i)
        words_new[-1] = "09"+iter+"_Brain-Glioma.png"
        words_new[-2] = "db_new"
        new_path = "\\".join(words_new)
        os.rename(item,new_path)
    for i,item in enumerate(meningioma):
        i = i+len(glioma)
        words_old = item.split('\\')
        words_new = words_old.copy()
        iter  = str(i)
        words_new[-1] = "09"+iter+"_Brain-Meningioma.png"
        words_new[-2] = "db_new"
        new_path = "\\".join(words_new)
        os.rename(item,new_path)
    for i,item in enumerate(good):
        i = i + len(glioma) +len(meningioma)
        words_old = item.split('\\')
        words_new = words_old.copy()
        iter  = str(i)
        words_new[-1] = "09"+iter+"_Brain-Good.png"
        words_new[-2] = "db_new"
        new_path = "\\".join(words_new)
        os.rename(item,new_path)
    for i,item in enumerate(pituitary):
        i = i + len(glioma) + len(meningioma) + len(pituitary)
        words_old = item.split('\\')
        words_new = words_old.copy()
        iter  = str(i)
        words_new[-1] = "09"+iter+"_Brain-Pituitary.png"
        words_new[-2] = "db_new"
        new_path = "\\".join(words_new)
        os.rename(item,new_path)
