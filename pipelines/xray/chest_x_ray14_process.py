import ast
import PIL
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



def main_chest_xray_14(source_dir, result_dir):
    target_path = os.path.join(result_dir, "Chest_X-ray14")
    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(target_path)
    # labels of dataset
    names = ["Healthy", "Atelectasis" , "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
    dir_s = source_dir          # path to source images
    dir_d = target_path          # destination path for images
    dir_labels = target_path+'/labels.csv'     # path to .csv file with labels

    for i in range(0, 12):
        if (i!=0):
            source1 = dir_s + '/images ' + str(i) + '/'
        else:
            source1 = dir_s + '/images' + '/'

        with open(dir_labels) as infile:
            lines = infile.readlines()[1:]
            # open file for each line in .csv file and process it
            for line in lines:
                f_name = line[:16]
                print(f_name)
                # open file
                try:
                    image = PIL.Image.open(source1 + f_name)
                    # extract diseases names from image description
                    labels = ast.literal_eval(line[17:])
                    labels = ast.literal_eval(labels)
                    diseases = [i for i in range(len(labels)) if labels[i] == 1]

                    n_name = f_name[:12]    # image identifier
                    # add diseases to file name
                    for d in diseases:
                        n_name += ('-' + names[d])
                    n_name += '.png'
                    # save with new name
                    image.save((dir_d + '/' + n_name), format="png")
                except IOError:
                    print('')



