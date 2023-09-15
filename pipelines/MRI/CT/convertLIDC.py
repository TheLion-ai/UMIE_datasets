import os
import glob
from pydicom import dcmread
import pydicom.pixel_data_handlers.util as ddh
import cv2
import numpy as np
import pandas as pd

def main_LIDC(source_dir, result_dir):

    target_path = os.path.join(result_dir, "LIDC")
    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(target_path)

    # CSV files that categorize size of nodule
    # Insert paths to csv file with SOP instance UID's of dicoms with nodules
    nodules = pd.read_csv((result_dir + '/nodules' + '.csv'), header=None).to_numpy()
    non_nodules = pd.read_csv((result_dir + '/non_nodules' + '.csv'), header=None).to_numpy()
    small_nodules = pd.read_csv((result_dir + '/small_nodules' + '.csv'), header=None).to_numpy()

    # Path to directory with images
    path = source_dir
    # Path to directory with masks
    path2 = os.path.join(target_path, "masks")
    # Extension of images
    extension = '.dcm'
    # Searching for images recursively
    imgs = glob.glob(f"{path}/**/*{extension}", recursive=True)
    msks = glob.glob(f"{path2}/**/*.png", recursive=True)
    print('Skończono globowanie')

    # Taking photos that have corresponding mask and creating malignancy dictionary
    sids = []
    malignancies = []
    for mask in msks:
        mask = mask.split('/')[-1]
        malignancy = mask.split('_')[-1][0]
        sid = mask.split('_')[0]
        sid = sid.replace('-', '')
        sids.append(sid)
        malignancies.append(malignancy)
    malignancyDict = {sids[i]: malignancies[i] for i in range(len(sids))}

    print(len(imgs))
    # DICOM conversion
    if extension == '.dcm':
        # iterate over found images
        for image in imgs:
            ds = dcmread(image)
            siuid = ds.SOPInstanceUID
            siuid = siuid.replace('.', '')


            foldername = True
            if siuid in sids:
                foldername = False

            siuid = siuid.replace('.', '')

            if (ds.Modality == 'CT'):

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

                WindowCenter = -550
                WindowWidth = 2000

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
                if foldername == True:
                    path11 = os.path.join(result_dir, "No Nodule")
                    if not os.path.exists(os.path.join(path11)):
                        os.makedirs(path11)
                    cv2.imwrite(path11 + f'/02{siuid}_Lungs-Good.png', windowed)
                else:
                    path12 = os.path.join(result_dir, "Nodule")
                    if not os.path.exists(os.path.join(path12)):
                        os.makedirs(path12)
                    cv2.imwrite(path12 + f'/02{siuid}_Lungs-Malignancy{malignancyDict[f"{siuid}"]}.png', windowed)
                print('zapisano obraz')

                # mark visited file as *.visited in case of failure it won't be visited again
                os.rename(image, image+'.visited')
                # if created .converted file remove it
                if (removeAtEnd):
                    os.system(f'rm {image}.converted')

    # # Extension of images
    extension = '.png'
    # # Searching for images recursively
    imgs = glob.glob(f"{path2}/**/*{extension}", recursive=True)

    # Rename to normalized name
    for image in imgs:
        mask = image.split('/')[-1]
        malignancy = mask.split('_')[-1][0]
        sid = mask.split('_')[0]
        sid = sid.replace('-', '')
        os.rename(image, f'{path}02{sid}_Lungs-Malignancy{malignancy}.png')
