import csv
import ast
import os

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




# This module is written for ChestX-ray14 dataset
def convert_labels(original_label, condition_names):
    # in original csv file there are more than single condition to one photo
    # and this function converts labels to binary vector [0 , 1, 0, 0 ...]
    original_label = original_label.lower()
    conditions = original_label.split('|')

    label = [0] * len(condition_names)
    for condition in conditions:
        try:
            label[condition_names.index(condition)] = 1
        except ValueError:
            print("Nie ma takiej zmainy na liscie")
    return label


def convert_csv_xray_14(source_dir, result_dir):

    target_path = os.path.join(result_dir, "Chest_X-ray14")
    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(target_path)
    condition_names = ["No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",  "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
    condition_names = [x.lower() for x in condition_names]
    data_csv = source_dir + '/other/Data_Entry_2017_v2020.csv'
    with open((target_path+'/labels.csv'), newline='', mode="w") as write_file:
        writer = csv.writer(write_file, delimiter=',')
        with open(data_csv, newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                img_id = row[0]
                img_label = convert_labels(row[1], condition_names)
                writer.writerow([img_id, img_label])

