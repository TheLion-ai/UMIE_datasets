import os
import glob
from pydicom import dcmread
import pydicom.pixel_data_handlers.util as ddh
import cv2
import numpy as np

def main_Brain_Tumor_Progression(source_dir, result_dir):

    target_path = os.path.join(result_dir, "Brain_Tumor_Progression")
    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(target_path)
    # path to directory with images
    path = source_dir
    # Path to directory with images
    # # Extension of images
    extension = '.dcm'
    # # Searching for images recursively
    imgs = glob.glob(f"{path}/**/*{extension}", recursive=True)
    print(imgs)


    # DICOM conversion
    if extension == '.dcm':
        # iterate over found images
        for image in imgs:
            ds = dcmread(image)
            desc = ds.SeriesDescription
            print(desc)
            patID = ds.PatientID
            date = ds.StudyDate

            # skipping files, where mask can't be applied
            if (desc == 'T1post') or (desc == 'Mask_Tumor'):

                removeAtEnd = False
                # if image is not little endian implicit convert using gdcmconv

                if (ds.is_little_endian == False):
                    # converting using gdcm conv save as nameOfFile.converted
                    os.system(f'gdcmconv -w -X -I -i {image} -o {image}.converted;')
                    ds = dcmread(f'{image}.converted')
                    # set property to remove *.converted file at the end
                    removeAtEnd = True

                # if window parameters are provided use it to remove redundant data
                array = ds.pixel_array
                # apply modality LUT
                windowed = ddh.apply_modality_lut(array, ds)

                WindowCenter = ds.WindowCenter
                WindowWidth = ds.WindowWidth

                # apply window
                windowed = np.clip(windowed, WindowCenter - WindowWidth / 2, WindowCenter + WindowWidth / 2)
                # convert from hounsfield scale (-1000 to 1000) to png scale (0 to 255)
                min = np.min(windowed)
                if min < -1000: min = -1000
                windowed = windowed - min
                maks = np.max(windowed)
                ratio = maks / 255
                windowed = np.divide(windowed, ratio).astype(int)
                # save image as nameOfFile.png
                if not os.path.exists(os.path.join(target_path,  f'{desc}')):
                    os.makedirs(os.path.join(target_path,  f'{desc}'))
                cv2.imwrite(os.path.join(target_path,  f'{desc}/10{patID}{date}{image[-6:-4]}_{desc}.png'), windowed)

                print('zapisano obraz')

                # if created .converted file remove it
                if (removeAtEnd):
                    os.system(f'rm {image}.converted')
