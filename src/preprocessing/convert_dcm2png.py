"""Convert dicom images to png format."""
import glob
import os

import cv2
import numpy as np
import pydicom
import pydicom.pixel_data_handlers.util as ddh
from pydicom import dcmread
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


# TODO: Add descriptions
class ConvertDcm2Png(BaseEstimator, TransformerMixin):
    """Convert dicom images to png format."""
    def __init__(
        self,
        window_width: int = None,
        window_center: int = None,
        on_error_remove: bool = True,
        **kwargs,
    ):
        self.window_width = window_width
        self.window_center = window_center
        self.on_error_remove = on_error_remove

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Convert dicom images to png format.
        Args:
            X (list): List of paths to the images.
        Returns:
            new_paths (list): List of paths to the converted images.
        """
        print("Converting dicom to png...")
        for img_path in tqdm(X):
            if img_path.endswith(".dcm"):
                self.convert_dcm2png(img_path)

        root_path = os.path.dirname(X[0])
        new_paths = glob.glob(f"{root_path}/*.png", recursive=True)
        return new_paths

    def convert_dcm2png(self, img_path):
        """Convert dicom images to png format.
        Args:
            img_path (str): Path to the image.
        """
        # iterate over found images
        ds = dcmread(img_path)
        ds = self._convert2little_endian(ds, img_path)
        try:
            output = ddh.apply_modality_lut(ds.pixel_array, ds)
            # if window parameters are provided use it to remove redundant data
            output = self._apply_window(output, ds)
            new_path = img_path.replace(".dcm", ".png")
            cv2.imwrite(new_path, output)

        except:
            # TODO: add logging
            print(f"Error occured while converting {img_path} {ds.is_little_endian}")
            if self.on_error_remove:
                os.remove(img_path)

    def _convert2little_endian(self, ds, img_path):
        """Convert dicom image to little endian.
        Args:
            ds (pydicom.dataset.FileDataset): Dicom image.
            img_path (str): Path to the image.
        Returns:
            ds (pydicom.dataset.FileDataset): Dicom image.
        """
        # if image is not little endian implicit convert using gdcmconv
        if ds.is_little_endian == False:
            # convert image to little endian
            os.system(f"gdcmconv -w -X -I -i {img_path} -o {img_path}.converted;")
            # Read converted image
            ds = dcmread(f"{img_path}.converted")
            # set property to remove *.converted file at the end
            os.system(f"rm {img_path}.converted")
        return ds

    def _get_window_parameters(self, ds):
        """Get window parameters from dicom image.
        Args:
            ds (pydicom.dataset.FileDataset): Dicom image.
        """
        # Sometimes window center is stored as a list of values, sometimes as a single value
        # If it is a list, we take the first value for simplicity
        if self.window_center is None:
            self.window_center = (
                int(ds.WindowCenter[0])
                if type(ds.WindowCenter) == pydicom.multival.MultiValue
                else int(ds.WindowCenter)
            )

        if self.window_width is None:
            self.window_width = (
                int(ds.WindowWidth[0])
                if type(ds.WindowWidth) == pydicom.multival.MultiValue
                else int(ds.WindowWidth)
            )

    def _apply_window(self, output, ds):
        """Apply window to the dicom image.
        Args:
            output (np.array): Dicom image.
            ds (pydicom.dataset.FileDataset): Dicom image.
        Returns:
            output (np.array): Dicom image.
        """
        self._get_window_parameters(ds)
        # apply window
        output = np.clip(
            output,
            self.window_center - self.window_width / 2,
            self.window_center + self.window_width / 2,
        )
        # convert from hounsfield scale (-1000 to 1000) to png scale (0 to 255)
        min = np.min(output)
        min = -1000 if min < -1000 else min
        output = output - min
        ratio = np.max(output) / 255
        output = np.divide(output, ratio).astype(int)
        return output
