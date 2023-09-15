import pydicom
import numpy as np
from pydicom import dcmread
import pydicom.pixel_data_handlers.util as ddh
import os
import cv2
import glob
from sklearn.base import BaseEstimator, TransformerMixin


class ConvertDcm2Png(BaseEstimator, TransformerMixin):

        def __init__(self):
                pass

        def fit(self, X, y=None):
                return self

        def transform(self,
                X,
                image_path,
                window_width = None,
                window_center=None
        ):
            # iterate over found images
            ds = dcmread(image_path)
        
            # if image is not little endian implicit convert using gdcmconv
        
            if (ds.is_little_endian == False):
                # converting using gdcm conv save as nameOfFile.converted
                os.system(f'gdcmconv -w -X -I -i {image_path} -o {image_path}.converted;')
                ds = dcmread(f'{image_path}.converted')
                # set property to remove *.converted file at the end
                os.system(f'rm {image_path}.converted')
        
            # if window parameters are provided use it to remove redundant data
            try:
                pixel_array = ds.pixel_array
                # apply modality LUT
                windowed = ds.apply_modality_lut(pixel_array, ds)
        
                if window_center is None:
                    if type(ds.WindowCenter) == pydicom.multival.MultiValue:
                        window_center = int(ds.WindowCenter[0])
                    else:
                        window_center = int(ds.WindowCenter)
                if window_width is None:
                    if type(ds.WindowWidth) == pydicom.multival.MultiValue:
                        window_width = int(ds.WindowWidth[0])
                    else:
                        window_width = int(ds.WindowWidth)
        
                # apply window
                windowed = np.clip(windowed, window_center - window_width / 2, window_center + window_width / 2)
                # convert from hounsfield scale (-1000 to 1000) to png scale (0 to 255)
                min = np.min(windowed)
                if min < -1000: min = -1000
                windowed = windowed - min
                maks = np.max(windowed)
                ratio = maks / 255
                windowed = np.divide(windowed, ratio).astype(int)
                # save image
                os.remove(image_path)
                cv2.imwrite(image_path.replace('.dcm', '.png'), windowed)
                return windowed
            except:
                print(f'Error occured while converting {image_path} {ds.is_little_endian}')        
