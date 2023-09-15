import cv2
import numpy as np
import glob
import os
import re

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
