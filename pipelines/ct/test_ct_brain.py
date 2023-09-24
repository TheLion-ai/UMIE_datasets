import PIL
import os
from PIL import Image
import csv

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

if __name__ == "__main__":
    dir_csv = ''  #csv file path
    dir_1 = ''  # images source directory
    dir_2 = ''   # masks source
    dir_d1 = ''     # images target directory
    dir_d2 = ''      # masks target directory
    dataset_id = ''   # dataset identifier

    with open(dir_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                folder = str(row[0])
                folder = folder.replace('\'', '')
                image = str(row[1])
                image = image.replace('\'', '')
                path_1 = dir_1 + folder + '/bone/' + image + '.jpg'
                path_2 = dir_2 + folder + '/brain/' + image + '.jpg'
                n_name = dataset_id
                if(row[7] == 1):
                    n_name = n_name + folder + image + '_Brain-Good.png'
                else:
                    n_name = n_name + folder + image + '_Brain-Hemorrhage.png'
                print(n_name)
                print(path_1)
                print(path_2)
                try:
                    image = PIL.Image.open(path_1)
                    mask = PIL.Image.open(path_2)
                    image.save((dir_d1 + n_name), format="png")
                    mask.save((dir_d2 + n_name), format="png")
                except IOError:
                    print('Not found image')

                line_count += 1
        print(f'Processed {line_count} lines.')


