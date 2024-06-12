"""Converts dicom files to png images with appropriate color encoding.""" ""
import glob
import os

import cv2
import numpy as np
import pydicom
import pydicom.pixel_data_handlers.util as ddh
from pydicom import dcmread
from base.step import BaseStep
from tqdm import tqdm
from base.step import BaseStep

class ConvertDcm2Png(BaseStep):
    """Converts dicom files to png images with appropriate color encoding."""



    def transform(self, X: list) -> list:
        """Convert dicom files to png images with appropriate color encoding.

        Args:
            X (list): List of paths to the images.
        Returns:
            X (list): List of paths to the images.
        """
        print("Converting dicom to png...")
        if len(X) == 0:
            raise ValueError("No list of files provided.")
        for img_path in tqdm(X):
            if img_path.endswith(".dcm"):
                self.convert_dcm2png(img_path)

        mask_paths = []
        for root, _, filenames in os.walk(self.masks_path):
            for filename in filenames:
                if filename.startswith("."):
                    continue
                elif filename.endswith(".dcm"):
                    path = os.path.join(root, filename)
                    # Verify if file is not a mask
                    if self.mask_selector in path:
                        mask_paths.append(path)
        if mask_paths:
            for mask_path in mask_paths:
                self.convert_dcm2png(mask_path)
        else:
            print("Masks not found.")

        root_path = self.source_path
        new_paths = [
            path
            for path in glob.glob(os.path.join(root_path, "**/*.png"), recursive=True)
            if self.mask_selector not in path
        ]
        return new_paths

    def convert_dcm2png(self, img_path: str) -> None:
        """Convert dicom files to png images with appropriate color encoding.

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

        except Exception as e:
            print(f"Error {e} occured while converting {img_path} {ds.is_little_endian}")
            if self.on_error_remove:
                os.remove(img_path)

    def _convert2little_endian(self, ds: pydicom.dataset.FileDataset, img_path: str) -> pydicom.dataset.FileDataset:
        """Convert dicom image to little endian.

        Args:
            ds (pydicom.dataset.FileDataset): Dicom file.
            img_path (str): Path to the image.
        Returns:
            ds (pydicom.dataset.FileDataset): Dicom file.
        """
        # if image is not little endian implicit convert using gdcmconv
        if ds.is_little_endian is False:
            # convert image to little endian
            os.system(f"gdcmconv -w -X -I -i {img_path} -o {img_path}.converted;")  # noqa: E702,E231
            # Read converted image
            ds = dcmread(f"{img_path}.converted")
            # set property to remove *.converted file at the end
            os.system(f"rm {img_path}.converted")
        return ds

    def _get_window_parameters(self, ds: pydicom.dataset.FileDataset) -> tuple:
        """Get window parameters from dicom file.

        Args:
            ds (pydicom.dataset.FileDataset): Dicom file.
        """
        # Sometimes window center is stored as a list of values, sometimes as a single value
        # If it is a list, we take the first value for simplicity
        if self.window_center is None:
            window_center = (
                int(ds.WindowCenter[0])
                if type(ds.WindowCenter) is pydicom.multival.MultiValue
                else int(ds.WindowCenter)
            )
        else:
            window_center = self.window_center

        if self.window_width is None:
            window_width = (
                int(ds.WindowWidth[0]) if type(ds.WindowWidth) is pydicom.multival.MultiValue else int(ds.WindowWidth)
            )
        else:
            window_width = self.window_width
        return window_center, window_width

    def _apply_window(self, output: np.ndarray, ds: pydicom.dataset.FileDataset) -> np.ndarray:
        """Apply window to the image.

        Args:
            output (np.ndarray): Image to apply window to.
            ds (pydicom.dataset.FileDataset): Dicom file.
        Returns:
            np.ndarray: Image data with applied window.
        """
        window_center, window_width = self._get_window_parameters(ds)
        # apply window
        output = np.clip(
            output,
            window_center - window_width / 2,
            window_center + window_width / 2,
        )
        # convert from hounsfield scale (-1000 to 1000) to png scale (0 to 255)
        min = np.min(output)
        min = -1000 if min < -1000 else min
        output = output - min
        ratio = np.max(output) / 255
        output = np.divide(output, ratio).astype(int)
        return output
