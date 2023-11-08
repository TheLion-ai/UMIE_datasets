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

def main_coronahack(source_dir, result_dir):

    target_path = os.path.join(result_dir, "Coronahack")
    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(target_path)
    dir = []    # all directories of source images
    dir.append(source_dir + '/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/')
    dir.append(source_dir + '/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/')
    dir_csv = source_dir + '/Chest_xray_Corona_Metadata.csv'    # path to .csv file with labels
    dir_d = target_path      # path to destination folder
    labels = ['Virus', 'bacteria']      # unified labels contained in dataset
    dataset_id = '05'   # identifier of dataset

    with open(dir_csv) as infile:
        lines = infile.readlines()[1:]
        # for each line in .csv file open, process and save image
        for line in lines:
            data1 = line.split(',')
            data1[len(data1)-1] = data1[len(data1)-1][:-1]
            # extract file name
            f_name = data1[1]
            # extract identifier of image without - and _
            namep = f_name.replace('_', '')
            namep = namep.replace('-', '')
            namep = namep.replace('.jpg', '')
            namep = namep.replace('.jpeg', '')
            n_name = dataset_id + namep + '_' + 'Lungs' + '-'       # firfile namest part of new
            print(n_name)
            for i in range(len(dir)):       # look for image in every source destination
                dir_1 = dir[i]
                # open image
                try:
                    image = PIL.Image.open(dir_1 + f_name)
                    label1 = data1[2]
                    # add appropriate label and save image
                    if(label1 == 'Normal'):
                        n_name += 'Good.png'
                        image.save((dir_d + '/' + n_name), format="png")
                    else:
                        label2 = data1[5]
                        if(label2 == 'Virus'):
                            n_name += 'PneumoniaVirus.png'
                            image.save((dir_d + '/' + n_name), format="png")
                        elif(label2 == 'bacteria'):
                            n_name += 'PneumoniaBacteria.png'
                            image.save((dir_d + '/' + n_name), format="png")
                except IOError:
                    print('N')
