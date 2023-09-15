import os
import glob
from pydicom import dcmread
import pydicom.pixel_data_handlers.util as ddh
import cv2
import numpy as np

def main_QIN_Brain(source_dir, result_dir):

    # Path to directory with images
    path = source_dir
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

            # skipping files, where mask can't be applied
            if desc != 'DSC':


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
                if not os.path.exists(result_dir + f'/Przetworzone_QIN-BRAIN-MRI/{desc}'):
                    os.makedirs(result_dir + f'/Przetworzone_QIN-BRAIN-MRI/{desc}')
                cv2.imwrite(result_dir + f'/Przetworzone_QIN-BRAIN-MRI/{desc}/{patID}{image[-6:-4]}_{desc}.png', windowed)

                print('zapisano obraz')

                if (removeAtEnd):
                    os.system(f'rm {image}.converted')
