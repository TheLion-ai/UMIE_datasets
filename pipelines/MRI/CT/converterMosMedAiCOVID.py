import os
import glob
import nibabel as nib
import cv2
import numpy as np
from pathlib import Path

def main_MosMed(source_dir, result_dir):

    target_path = os.path.join(result_dir, "MosMed")
    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(target_path)

    extension = '.gz'

    # Path to directory with images
    path = source_dir

    # Searching for images recursively
    files = glob.glob(f"{path}/**/*{extension}", recursive=True)

    # iterate over found nii files
    for file in files:
        nii_img = nib.load(file)
        nii_data = nii_img.get_fdata()

        slices = nii_data.shape[2]

        pat0 = Path(file).resolve().parent
        pat1 = pat0.parent
        categoryname = str(pat0)[len(str(pat1)) + 1:]

        filename = file[len(str(pat0))+1:]

        for slc in range(slices):
            image = np.array(nii_data[:, :, slc])
            pat0 = Path(file).resolve().parent
            pat1 = pat0.parent
            # name of directory containing nii file
            categoryname = str(pat0)[len(str(pat1)) + 1:]

            WindowCenter = -550
            WindowWidth = 2000
            # apply window
            windowed = np.clip(image, WindowCenter - WindowWidth / 2, WindowCenter + WindowWidth / 2)

            # convert from hounsfield scale (-1000 to 1000) to png scale (0 to 255)
            min = np.min(windowed)
            windowed = windowed - min
            maks = np.max(windowed)
            ratio = maks / 255
            windowed = np.divide(windowed, ratio).astype(int)

            if categoryname == 'masks':
                category = 'Masks'
                label = 'Parenchyma0025'
            # Choosing category name based on folder it's contained in
            else:
                category = 'Images'
                if categoryname == 'CT-0':
                    label = 'Good'
                elif categoryname == 'CT-1':
                    label = 'Parenchyma0025'
                elif categoryname == 'CT-2':
                    label = 'Parenchyma2550'
                elif categoryname == 'CT-3':
                    label = 'Parenchyma5075'
                elif categoryname == 'CT-4':
                    label = 'Parenchyma7500'


            # save image as nameOfFile.png
            if not os.path.exists(os.path.join(target_path, categoryname)):
                os.makedirs(os.path.join(target_path, categoryname))
            cv2.imwrite(os.path.join(target_path, categoryname)+f'/01{filename[6:10]}{slc}_Lungs-{label}.png', windowed)
