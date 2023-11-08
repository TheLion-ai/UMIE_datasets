import cv2
import numpy as np
import glob
import os
import re

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



def transfer(image,color):
    #this part lests us change the color of non black pixels of an image
    im = np.array(image)
    image_copy = im.copy()
    black_pixels_mask = np.all(im == [0, 0, 0], axis=-1)
    non_black_pixels_mask = np.any(im != [0, 0, 0], axis=-1)
    image_copy[black_pixels_mask] = [0, 0, 0]
    image_copy[non_black_pixels_mask] = [color, color, color]
    return image_copy

def main_QIN_Brain_segmentation(source_dir, result_dir):
    #dir to database
    dir = result_dir + f'/Przetworzone_QIN-BRAIN-MRI/'
    #names of the folders of our database
    dirnames = ["AIFx3","Mask_Brain","Mask_NACC","Mask_NAWM","Mask_Tumor","Anatomic"]
    target_path = os.path.join(result_dir, "Przetworzone_QIN-BRAIN-MRI")
    target_path = os.path.join(target_path, "masks")
    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(target_path)
    savedir_to_masks = target_path
    target_path = os.path.join(result_dir, "Przetworzone_QIN-BRAIN-MRI")
    target_path = os.path.join(target_path, "images")
    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(target_path)
    savedir_to_images = target_path
    anatomic = []
    aif  = []
    brain = []
    nacc = []
    nawm = []
    tumor = []
    #sorting images into lists
    for i,folder in enumerate(dirnames):
        path = dir + folder
        for file in glob.glob(path + "\\*.png"):
            if i == 0:
                aif.append(file)
            elif i == 1:
                brain.append(file)
            elif i == 2:
                nacc.append(file)
            elif i == 3:
                nawm.append(file)
            elif i == 4:
                tumor.append(file)
            elif i == 5:
                anatomic.append(file)
    filenames = []
    #iterating for unifying names
    for file in anatomic:
        words = file.split("\\")
        word = words[-1]
        temp = re.split('-|_',word)
        prefix  =  temp[:4]
        ID  =  temp[4:6]
        joined  =  "-".join(prefix) +"-" + "-".join(ID) + "_Brain"
        filenames.append(joined)
    #here we check whether the masks represents a healthy brain or not
    for i in range(len(tumor)):
        im = cv2.imread(tumor[i])
        arr = np.array(im)
        is_all_zero = np.all((arr == 0))
        if is_all_zero:
            filenames[i] = filenames[i] + "-Good.png"
        else:
            filenames[i] = filenames[i] + "-Tumor.png"

    aif_trans = []
    brain_trans = []
    nacc_trans = []
    nawm_trans = []


    for i in range(len(tumor)):
        #changing the brain mask
        image = cv2.imread(brain[i])
        image_copy = transfer(image,50)
        brain_trans.append(image_copy)
        #changing the nawm mask
        image = cv2.imread(nawm[i])
        image_copy = transfer(image, 200)
        nawm_trans.append(image_copy)
        #changing the nacc mask
        image = cv2.imread(nacc[i])
        image_copy = transfer(image, 180)
        nacc_trans.append(image_copy)
        #changing the aif mask
        image = cv2.imread(aif[i])
        image_copy = transfer(image, 190)
        aif_trans.append(image_copy)
    shape_im  = aif_trans[0].shape

    for i in  range(len(tumor)):
        #this allows us to merge all the masks into one
        non_black_pixels_mask = np.any(nawm_trans[i] != [0, 0, 0], axis=-1)
        brain_trans[i][non_black_pixels_mask] = nawm_trans[i][non_black_pixels_mask]
        non_black_pixels_mask = np.any(nacc_trans[i] != [0, 0, 0], axis=-1)
        brain_trans[i][non_black_pixels_mask] = nacc_trans[i][non_black_pixels_mask]
        non_black_pixels_mask = np.any(aif_trans[i] != [0, 0, 0], axis=-1)
        brain_trans[i][non_black_pixels_mask] = aif_trans[i][non_black_pixels_mask]
        im  = cv2.imread(tumor[i])
        non_black_pixels_mask = np.any(im != [0, 0, 0], axis=-1)
        brain_trans[i][non_black_pixels_mask] = im[non_black_pixels_mask]
    os.chdir(savedir_to_masks)
    #saving to 2 separate folders
    for i in range(len(tumor)):
        filename = filenames[i]
        os.chdir(savedir_to_masks)
        cv2.imwrite(filename,brain_trans[i])
        image  =  cv2.imread(anatomic[i])
        os.chdir(savedir_to_images)
        cv2.imwrite(filename,image)
