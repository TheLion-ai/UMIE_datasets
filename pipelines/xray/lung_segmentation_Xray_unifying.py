# This is module to uniform Lung segmentation from Chest X-Rays
# to use this module you just need to change global paths to dataset, and the rest will be done automatically
# not every image has a mask, so our base is masks names, and we discard images with no masks
import cv2 as cv
import os
from os import listdir
from os.path import isfile, join
from shutil import copyfile

# # paths to dataset original images
# masks_path = "dataset_lungs/data/Lung Segmentation/masks"
# images_path = "dataset_lungs/data/Lung Segmentation/CXR_png"
#
# # all modified masks, we discard MCUCXR with abnormalities because they are not well defined
# modified_masks_dir = "modified_masks"
#
# # in those folders we will have our prepared masks/images, we also discard images with not specified labels
# uniformed_masks_dir = "uniformed_masks"
# uniformed_images_dir = "uniformed_images"


# this function removes 'mask' from file name
# this is useful when looking for original images
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


def img_name_from_mask(mask_name):
    split_mask = mask_name.split('_')
    split_ext = split_mask[-1].split('.')
    s = "_"
    image_name = ''
    if split_mask[0] == 'CHNCXR':
        image_name = s.join(split_mask[:3]) + ".png"
    elif split_mask[0] == 'MCUCXR':
        image_name = s.join(split_mask[:2]) + "_" + split_ext[0] + ".png"
    else:
        pass
    return image_name


# Takes masks, converts it 3channels -> 1channel and changes 255 as white pixels
# to 170 which is the value for lungs in our segmentation
def prepare_masks(mask_path, modified_masks_path):

    masks = [f for f in listdir(mask_path) if isfile(join(mask_path, f))]

    if not os.path.exists(modified_masks_path):
        os.mkdir(modified_masks_path)
        for mask_name in masks:
            mask = cv.imread(join(mask_path, mask_name))
            img_gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
            ret, thresh1 = cv.threshold(img_gray, 254, 170, cv.THRESH_BINARY)
            cv.imwrite(join(modified_masks_path, mask_name), thresh1)


# images in MCUCXR with abnormalities are not well defined, so we discard them
def select_images(name):
    split_mask = name.split('_')
    split_ext = split_mask[-1].split('.')
    is_valid = False
    if split_mask[0] == "MCUCXR" and split_ext[0] == '1':
        is_valid = False
    else:
        is_valid = True

    return is_valid


# module to uniform names convention: prefix + dataset_id + body_part + -label.png"
# you can change it, so it goes in tandem with your dataset
def uniformed_name(name):
    dataset_prefix = "08"
    split_name = name.split('_')
    split_ext = split_name[2].split('.')

    if split_name[0] == 'CHNCXR' and split_ext[0] == '1':
        name = dataset_prefix + split_name[0] + split_name[1] + '_Lungs-' + "Tuberculosis.png"
    elif split_name[0] == 'CHNCXR':
        name = dataset_prefix + split_name[0] + split_name[1] + '_Lungs-' + "Good.png"
    elif split_name[0] == "MCUCXR" and split_ext[0] == '0':
        name = dataset_prefix + split_name[0] + split_name[1] + '_Lungs-' + "Good.png"
    else:
        pass

    return name


# main function to uniform dataset.
# based on masks we create list of file names
# we remove 'mask' from names to find original images
# in our convention masks and images have the same name but they are in separate folders
def uniform_dataset(images_path, modified_masks_dir, uniformed_masks_dir, uniformed_images_dir):
    masks = [f for f in listdir(modified_masks_dir) if isfile(join(modified_masks_dir, f))]

    if not os.path.exists(uniformed_masks_dir):
        os.mkdir(uniformed_masks_dir)

    if not os.path.exists(uniformed_images_dir):
        os.mkdir(uniformed_images_dir)

    for mask in masks:
        img_name = img_name_from_mask(mask)
        if select_images(img_name):
            img_uniformed_name = uniformed_name(img_name)
            copyfile(join(images_path, img_name), join(uniformed_images_dir, img_uniformed_name))
            copyfile(join(modified_masks_dir, mask), join(uniformed_masks_dir, img_uniformed_name))
        else:
            pass


def main_lung_segmentation(source_dir, result_dir):
    # paths to dataset original images
    masks_path = source_dir + "/masks"
    images_path = source_dir + "/CXR_png"

    target_path = os.path.join(result_dir, "Lung_Segmentation_X_ray")
    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(target_path)
    # all modified masks, we discard MCUCXR with abnormalities because they are not well defined
    modified_masks_dir = os.path.join(target_path, "modified_masks")
    if not os.path.exists(os.path.join(modified_masks_dir)):
        os.makedirs(modified_masks_dir)

    # in those folders we will have our prepared masks/images, we also discard images with not specified labels
    uniformed_masks_dir = os.path.join(target_path, "uniformed_masks")
    if not os.path.exists(os.path.join(uniformed_masks_dir)):
        os.makedirs(uniformed_masks_dir)
    uniformed_images_dir = os.path.join(target_path, "uniformed_images")
    if not os.path.exists(os.path.join(uniformed_images_dir)):
        os.makedirs(uniformed_images_dir)

    prepare_masks(masks_path, modified_masks_dir)
    uniform_dataset(images_path, modified_masks_dir, uniformed_masks_dir, uniformed_images_dir)

