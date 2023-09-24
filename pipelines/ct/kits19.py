import os
import glob
import json
import nibabel as nib
import cv2
import numpy as np

import yaml
import os
from preprocessing.get_file_paths import GetFilePaths
from preprocessing.add_new_ids import AddNewIds
from preprocessing.convert_nii2png import ConvertNii2Png
from preprocessing.convert_dcm2png import ConvertDcm2Png
from preprocessing.create_file_tree import CreateFileTree
from preprocessing.create_blank_masks import CreateBlankMasks
from preprocessing.delete_imgs_without_masks import DeleteImgsWithoutMasks
from preprocessing.copy_png_masks import CopyPNGMasks
from preprocessing.recolor_masks import RecolorMasks
from preprocessing.add_labels import AddLabels
from sklearn.pipeline import Pipeline

# TODO: create task config

def preprocess_kits19(source_path, target_path, labels_path):
    dataset_name = "KITS-19"
    dataset_uid = yaml.load(open('config/dataset_uid_config.yaml'), Loader=yaml.FullLoader)[dataset_name] 
    phases = yaml.load(open('config/phases_config.yaml'), Loader=yaml.FullLoader)[dataset_name]
    mask_encoding_config = yaml.load(open('config/masks_encoding_config.yaml'), Loader=yaml.FullLoader)
    dataset_masks = yaml.load(open('config/dataset_masks_config.yaml'), Loader=yaml.FullLoader)[dataset_name]

    mask_colors_old2new = {v: mask_encoding_config[k] for k, v in dataset_masks.items()}


    def get_label_kits19(img_path, mask_folder_name="Masks", kidney_tumor_encoding=101):
                img_id = os.path.basename(img_path)
                root_path = os.path.dirname(os.path.dirname(img_path))
                mask_path = os.path.join(root_path, mask_folder_name, img_id)
                mask = cv2.imread(mask_path)
                if kidney_tumor_encoding in np.unique(mask):
                    study_id = os.path.basename(os.path.dirname(img_path))
                    labels = []
                    for case in labels_list:
                        if case['case_id'] == study_id:
                            label = case['tumor_histologic_subtype'].replace('_', '')
                            labels.append(label)
                    return labels

    params = {
        "target_path": target_path,
        "labels_path": labels_path,
        "dataset_name": dataset_name,
        "dataset_uid": dataset_uid,
        "phases": phases,
        "dataset_masks": dataset_masks,
        "z-fill": 2,
        "img_id_extractor": lambda x: os.path.basename(x).split("_")[-1],
        "study_id_extractor": lambda x: os.path.basename((os.path.dirname(x))).split("_")[-1],
        "mask_colors_old2new": mask_colors_old2new,
        "window_center": 50, # TODO: add to config
        "window_width": 400,
        "mask_selector": "segmentation",
        "get_label": get_label_kits19,
    }

    pipeline = Pipeline(steps=[
        ("create_file_tree", CreateFileTree(**params)),
        ("get_file_paths", GetFilePaths(**params)),
        ("convert_nii2Png", ConvertNii2Png(**params)), # TODO: add conversion from nii to dcm
        ("copy_png_masks", CopyPNGMasks(**params)),
        ("add_new_ids", AddNewIds(**params)),
        ("recolor_masks", RecolorMasks(**params)),
        ("add_labels", AddLabels(**params)),
        # # Choose either to create blank masks or delete images without masks
        # # ("create_blank_masks", CreateBlankMasks(**params)),
        # ("delete_imgs_without_masks", DeleteImgsWithoutMasks(**params))
    ], )



    pipeline.transform(X=source_path)


### old code    

    target_path = os.path.join(result_dir, "KITS19")
    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(target_path)

    extension = '.gz'

    # Path to directory with images
    path = source_dir

    # Searching for images recursively
    files = glob.glob(f"{path}/**/*{extension}", recursive=True)

    # iterate over found nii files (Masks)
    caseSlices = []
    labels = []

    for file in files:
        nii_img = nib.load(file)
        nii_data = nii_img.get_fdata()
        filetype = file.split('/')[-1][:-7]
        caseNumber = file.split('/')[-2][-5:]

        if filetype == 'segmentation':
            slices = nii_data.shape[0]

            for slc in range(slices):
                # creating id for image
                caseSlice = f'{caseNumber}{slc}'
                caseSlices.append(caseSlice)

                image = np.array(nii_data[slc, :, :])
                # Applying window
                minimal = np.min(image)
                windowed = image - minimal
                maks = np.max(windowed)
                ratio = maks / 255
                windowed = np.divide(windowed, ratio).astype(int)
                if len(np.unique(windowed)) == 3:
                    label = 'Tumor'
                else:
                    label = 'Good'
                    windowed = np.clip(windowed, 0, 127)

                labels.append(label)

                # save image as nameOfFile.png
                if not os.path.exists(target_path+'/Masks'):
                    os.makedirs(target_path+'/Masks')
                cv2.imwrite(target_path+'/Masks'+f'/00{caseSlice}_Kidneys-{label}.png', windowed)
    # creating dictionary for images of certain ids
    labelDict = {caseSlices[i]: labels[i] for i in range(len(caseSlices))}

    # iterating over images and coverting to png
    for file in files:
        nii_img = nib.load(file)
        nii_data = nii_img.get_fdata()
        filetype = file.split('/')[-1][:-7]
        caseNumber = file.split('/')[-2][-5:]

        if filetype == 'imaging':
            slices = nii_data.shape[0]

            for slc in range(slices):

                caseSlice = f'{caseNumber}{slc}'

                image = np.array(nii_data[slc, :, :])

                WindowCenter = 50
                WindowWidth = 400
                # apply window
                windowed = np.clip(image, WindowCenter - WindowWidth / 2, WindowCenter + WindowWidth / 2)

                minimal = np.min(windowed)
                windowed = windowed - minimal
                maks = np.max(windowed)
                ratio = maks / 255
                windowed = np.divide(windowed, ratio).astype(int)

                # save image as nameOfFile.png
                if not os.path.exists(target_path+'/Images'):
                    os.makedirs(target_path+'/Images')
                cv2.imwrite(target_path+'/Images'+f'/00{caseSlice}_Kidneys-{labelDict[f"{caseSlice}"]}.png', windowed)

    # path to directory with images
    with open(source_dir+'kits.json') as json_file:
        data = json.load(json_file)
    # # Extension of images
    extension = '.png'
    # # Searching for images recursively
    imgs = glob.glob(f"{path}/**/*{extension}", recursive=True)

    # Rename to normalized name
    for image in imgs:
        filename = image.split('\\')[-1]
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        labelT = label.split('-')[-1]
        caseid = filename[2:7]
        caseid = f'case_{caseid}'
        print(caseid)
        if labelT == 'Tumor':
            for case in data:
                if case['case_id'] == caseid:
                    newlabelT = case['tumor_histologic_subtype'].replace('_', '')
                    os.rename(image, f'{image.split("-")[0]}-{newlabelT}.png')

    #changing masks tumor values to certain values
    # # Searching for images recursively
    imgs = glob.glob(f"{path}/**/*{extension}", recursive=True)

    for image in imgs:
        filename = image.split('\\')[-1]
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        labelT = label.split('-')[-1]
        if labelT != 'Good':
            tumorvalue = 101
            img = cv2.imread(image)
            # changing pixel values
            np.place(img, img == 255, tumorvalue)
            cv2.imwrite(image, img)
            print(tumorvalue)
