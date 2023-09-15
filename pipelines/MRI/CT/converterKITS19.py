import os
import glob
import json
import nibabel as nib
import cv2
import numpy as np

def main_KITS19(source_dir, result_dir):

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
