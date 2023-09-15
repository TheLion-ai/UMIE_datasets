import os
import glob
import nibabel as nib
import cv2
import numpy as np
from pathlib import Path

def main_CT_ORG(source_dir, result_dir):

    target_path = os.path.join(result_dir, "CT-ORG")
    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(target_path)

    extension = '.gz'

    # Path to directory with images
    path = source_dir

    # Searching for images recursively
    files = glob.glob(f"{path}/**/*{extension}", recursive=True)

    dictlabels = {150: 'Liver', 160: 'Bladder', 170: 'Lungs', 127: 'Kidneys', 140: 'Bone', 50: 'Brain', 0: 'Nothing'}

    numbersslice = []
    partofbody = []

    # iterate over found nii files
    for file in files:

        pat0 = Path(file).resolve().parent
        pat1 = pat0.parent
        categoryname = str(pat0)[len(str(pat1)) + 1:]

        filename = file[len(str(pat0)) + 1:]

        # Iterating over masks And creating dictionary of labels based on most appearing part of body
        if categoryname == 'Masks':
            nii_img = nib.load(file)
            nii_data = nii_img.get_fdata()

            slices = nii_data.shape[2]
            for slc in range(slices):
                # getting number of nifti file
                number = filename.split('-')
                number = number[1].split('.')
                number = number[0]

                image = np.array(nii_data[:, :, slc], dtype=int)
                # Changing given mask levels to ones specified in google doc
                np.place(image, image == 1, 150)
                np.place(image, image == 2, 160)
                np.place(image, image == 3, 170)
                np.place(image, image == 4, 127)
                np.place(image, image == 5, 140)
                np.place(image, image == 6, 50)

                # Checking ocurance of certain body parts
                lbls, counts = np.unique(image, return_counts=True)

                if(len(lbls) != 1):
                    lbls = lbls[1:]
                    counts = counts[1:]

                label = dictlabels[lbls[np.argmax(counts)]]

                # # save image as nameOfFile.png
                if label != 'Nothing':
                    if not os.path.exists(target_path+f'{categoryname}'):
                        os.makedirs(target_path+f'{categoryname}')
                    cv2.imwrite(target_path+f'{categoryname}/13{number}s{slc}_{label}.png', image)

                numbersslice.append(f'{number}s{slc}')
                partofbody.append(label)

                print('Zapisano maskę')

    # creating dictionary of labels
    labels = {numbersslice[i]: partofbody[i] for i in range(len(numbersslice))}
    # creating dictionary of windows for certain body parts
    windowcenters = {'Liver': 30, 'Bladder': 50, 'Lungs': -550, 'Kidneys': 50, 'Bone': 1800, 'Brain': 40}
    windowwidths = {'Liver': 150, 'Bladder': 400, 'Lungs': 2000, 'Kidneys': 400, 'Bone': 400, 'Brain': 80}
    # iterate over found nii image files
    for file in files:

        pat0 = Path(file).resolve().parent
        pat1 = pat0.parent
        categoryname = str(pat0)[len(str(pat1)) + 1:]

        filename = file[len(str(pat0)) + 1:]

        if categoryname == 'imgs':
            nii_img = nib.load(file)
            nii_data = nii_img.get_fdata()

            slices = nii_data.shape[2]
            for slc in range(slices):
                number = filename.split('-')
                number = number[1].split('.')
                number = number[0]

                image = np.array(nii_data[:, :, slc], dtype=int)
                # getting label from dictionary
                label = labels[f'{number}s{slc}']

                if label != 'Nothing':

                    WindowCenter = windowcenters[label]
                    WindowWidth = windowwidths[label]
                    # apply window
                    windowed = np.clip(image, WindowCenter - WindowWidth / 2, WindowCenter + WindowWidth / 2)

                    min = np.min(windowed)
                    windowed = windowed - min
                    maks = np.max(windowed)
                    ratio = maks / 255
                    windowed = np.divide(windowed, ratio).astype(int)


                    # # save image as nameOfFile.png
                    if not os.path.exists(os.path.join(target_path, categoryname)):
                        os.makedirs(os.path.join(target_path, categoryname))
                    cv2.imwrite(os.path.join(target_path, categoryname)+'/13'+str(number)+str(slc)+str(label) +'.png', windowed)

                    print('Zapisano obrazek')
