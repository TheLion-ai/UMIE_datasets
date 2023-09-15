"""Change file names to match the format of the rest of the dataset."""
import os
import glob
import shutil
import re

import yaml
import cv2


def add_new_ids(
        source_path: str,
        target_path: str,
        dataset_name: str, 
        image_folder_name: str = 'Images',
        mask_folder_name: str = 'Masks'
        ):
    """
    Change file names to match the format of the rest of the dataset.

    Args:
        source_path (str): path to the root folder with masks and images
        target_path (str): path to the folder where recolored masks will be saved
        dataset_name (str): name of the dataset to be renamed, check dataset config for the list of available uids for datasets and add new ones if needed
    """
    dataset_uid_config = yaml.load(open('config/dataset_uid_config.yaml'), Loader=yaml.FullLoader)
    dataset_uid  = dataset_uid_config[dataset_name]
    phases = yaml.load(open('config/phases_config.yaml'), Loader=yaml.FullLoader)[dataset_name]


    if not os.path.exists(target_path):
        os.makedirs(target_path)
    # for phase in phases.values():
    #     if not os.path.exists(os.path.join(target_path, phase)):
    #         os.makedirs(os.path.join(target_path, phase))
    #     if not os.path.exists(os.path.join(target_path, phase, 'Images')):
    #         os.makedirs(os.path.join(target_path, phase, 'Images'))
    #     if not os.path.exists(os.path.join(target_path, phase, 'Masks')):
    #         os.makedirs(os.path.join(target_path, phase, 'Masks'))

    
    
    folders = ["Images", "Masks"]
    # Searching for images recursively
    # Even for classification dataset, we create an empty masks folder
    # We assume that the images and masks for the same patient are in one folder
    # each patients folder is named with the patient id
    # Each patient folder can contain subfolders with images from different phases and a separate folder for masks
    # We keep only the key phase
    
    # Stanford Brain
    # for study_id in os.listdir(source_path):
    #     if study_id.startswith('.'):
    #         continue
    #     for phase in phases.keys():
    #         for file in os.listdir(os.path.join(source_path, study_id, phase)):
    #             file_name = f"{dataset_uid}{phase}{study_id}{os.path.basename(file)}"
    #             folders = ["Images", "Masks"]
    #             for folder in folders:
    #                 new_path = os.path.join(target_path, phases[phase], folder, file_name)
    #                 if not os.path.exists(new_path):
    #                     # open image and save it with a different name
    #                     if folder == "Images":
    #                         source_file = os.path.join(source_path, study_id, phase, file)
    #                         shutil.copy2(source_file, new_path)
    #                         # img = cv2.imread(os.path.join(source_path, study_id, phase, file))
    #                         # img = cv2.imread(os.path.join(source_path, study_id, phase, file))
    #                         # cv2.imwrite(new_path, img)
    #                     else:
    #                         # mask = cv2.imread(os.path.join(source_path, study_id, mask_folder_name, file))
    #                         # cv2.imwrite(new_path, mask)
    #                         source_file = os.path.join(source_path, study_id, mask_folder_name, file)
    #                         shutil.copy2(source_file, new_path)

    # Stanford COCA
    if not os.path.exists(os.path.join(target_path, 'Images')):
        os.makedirs(os.path.join(target_path, 'Images'))
    if not os.path.exists(os.path.join(target_path, 'Masks')):
        os.makedirs(os.path.join(target_path, 'Masks'))
    for study_id in os.listdir(os.path.join(source_path, image_folder_name)):
        if study_id.startswith('.'):
            continue
        path = os.path.join(source_path, image_folder_name, study_id)
        extension = ".dcm"
        files = glob.glob(f"{path}/**/*{extension}", recursive=True)
        for file in files:
            basename = re.findall(r'[^-]*$', os.path.basename(file))[0]
            file_name = f"{dataset_uid}{study_id}{basename}"
            folders = ["Images", "Masks"]
            for folder in folders:
                new_path = os.path.join(target_path, folder, file_name)
                if not os.path.exists(new_path):
                    # open image and save it with a different name
                    if folder == "Images":
                        # source_file = os.path.join(source_path, study_id, phase, file)
                        shutil.copy2(file, new_path)
                    else:
                        pass
                    # else:
                    #     # source_file = os.path.join(source_path, study_id, mask_folder_name, file)
                    #     shutil.copy2(file, new_path)

