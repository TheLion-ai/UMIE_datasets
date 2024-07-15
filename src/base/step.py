"""Change img ids to match the format of the rest of the dataset."""
import os
from typing import Callable, Optional

import numpy as np
from sklearn.base import TransformerMixin

from base.extractors.img_id import BaseImgIdExtractor
from config.dataset_config import MaskColor
from constants import IMG_FOLDER_NAME, MASK_FOLDER_NAME


class BaseStep(TransformerMixin):
    """Change img ids to match the format of the rest of the dataset."""

    def __init__(
        self,
        source_path: str,  # path to the dataset that is being processed
        target_path: str,  # path to the directory where the processed dataset will be saved path to the source masks file
        dataset_uid: str,  # unique identifier for the dataset
        dataset_name: str,  # name of the dataset
        phases: dict[
            str, str
        ],  # phase_id used for encoding the phase in img name, phase_name used for naming the folder
        image_folder_name: str = IMG_FOLDER_NAME,  # name of folder, where images will be stored
        mask_folder_name: str = MASK_FOLDER_NAME,  # name of folder, where masks will be stored
        img_id_extractor: BaseImgIdExtractor = BaseImgIdExtractor(),  # function to extract image id from the image path
        study_id_extractor: Callable = lambda x: x,  # function to extract study id from the image path
        phase_id_extractor: Callable = lambda x: "0",  # function to extract phase from the image path
        zfill: Optional[int] = None,  # number of digits to pad the image id with
        window_center: Optional[int] = None,  # value used to process DICOM images
        window_width: Optional[int] = None,  # value used to process DICOM images
        label_extractor: Optional[Callable] = None,  # function to get label for the individual image
        img_prefix: Optional[str] = None,  # prefix of the source image file names
        segmentation_prefix: Optional[str] = None,  # prefix of the source mask file names
        mask_selector: Optional[str] = None,  # string included only in masks names
        multiple_masks_selector: Optional[dict] = None,
        labels: dict[str, list[dict[str, float]]] = {},  # some labels have multiple RadLex codes
        masks: dict[str, MaskColor] = {},
        labels_path: Optional[str] = None,  # path to the labels file
        masks_path: Optional[str] = None,  #
    ):
        """
        Initialize a Step object.

        Args:
            source_path (str): Path to the dataset that is being processed.
            target_path (str): Path to the directory where the processed dataset will be saved.
            labels_path (Optional[str]): Path to the labels file.
            masks_path (Optional[str]): Path to the source masks file.
            dataset_uid (str): Unique identifier for the dataset.
            dataset_name (str): Name of the dataset.
            phases (dict[str, str]): Dictionary containing phase_id used for encoding the phase in img name and phase_name used for naming the folder.
            image_folder_name (str, optional): Name of the folder where images will be stored. Defaults to IMG_FOLDER_NAME.
            mask_folder_name (str, optional): Name of the folder where masks will be stored. Defaults to MASK_FOLDER_NAME.
            img_id_extractor (BaseImgIdExtractor, optional): Function to extract image id from the image path. Defaults to BaseImgIdExtractor().
            study_id_extractor (Callable, optional): Function to extract study id from the image path. Defaults to lambda x: x.
            phase_id_extractor (Callable, optional): Function to extract phase from the image path. Defaults to lambda x: "0".
            zfill (Optional[int], optional): Number of digits to pad the image id with. Defaults to None.
            window_center (Optional[int], optional): Value used to process DICOM images. Defaults to None.
            window_width (Optional[int], optional): Value used to process DICOM images. Defaults to None.
            label_extractor (Optional[Callable], optional): Function to get label for the individual image. Defaults to None.
            img_prefix (Optional[str], optional): Prefix of the source image file names. Defaults to None.
            segmentation_prefix (Optional[str], optional): Prefix of the source mask file names. Defaults to None.
            mask_selector (Optional[str], optional): String included only in masks names. Defaults to None.
            multiple_masks_selector (Optional[dict], optional): Dictionary containing multiple masks selectors. Defaults to None.
            labels (dict[str, list[dict[str, float]]], optional): Dictionary containing labels with multiple RadLex codes. Defaults to {}.
            masks (dict[str, MaskColor], optional): Dictionary containing masks. Defaults to {}.
        """
        self.target_path = target_path
        self.dataset_name = dataset_name
        self.dataset_uid = dataset_uid
        self.phases = phases
        self.image_folder_name = image_folder_name
        self.mask_folder_name = mask_folder_name
        self.img_id_extractor = img_id_extractor
        self.study_id_extractor = study_id_extractor
        self.phase_id_extractor = phase_id_extractor
        self.segmentation_prefix = segmentation_prefix
        self.paths_data = np.array([])
        self.source_path = source_path
        self.labels_path = labels_path
        self.masks_path = masks_path
        self.zfill = zfill
        self.window_center = window_center
        self.window_width = window_width
        self.label_extractor = label_extractor
        self.img_prefix = img_prefix
        self.mask_selector = mask_selector
        self.multiple_masks_selector = multiple_masks_selector
        self.labels = labels
        self.masks = masks
        self.json_path = os.path.join(
            self.target_path,
            f"{self.dataset_uid}_{self.dataset_name}",
            f"{self.dataset_uid}_{self.dataset_name}.jsonl",
        )

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Change img ids to match the format of the rest of the dataset.

        Args:
            X (list): List of image paths.
        """
        raise NotImplementedError("This method should be implemented in the derived class.")

    def get_umie_id(self, img_path: str) -> str:
        """Create a unique identifier for the image.

        Args:
            img_path (str): Path to the image.

        Returns:
            str: Unique identifier for the image.
        """
        img_id = self.img_id_extractor(img_path)
        ext = os.path.splitext(img_path)[1]
        img_id = img_id.replace(ext, ".png")

        study_id = self.study_id_extractor(img_path)
        phase_id = self.phase_id_extractor(img_path)

        umie_id = f"{self.dataset_uid}_{phase_id}_{study_id}_{img_id}"
        return umie_id

    def validate_umie_path(self, img_path: str) -> bool:
        """Check if umie_path is a valid path (a mechanism for discarding imgs).

        Args:
            img_path (str): Path to the image.

        Returns:
            bool: If true, the path is validate.
        """
        img_id = self.img_id_extractor(img_path)
        study_id = self.study_id_extractor(img_path)
        phase_id = self.phase_id_extractor(img_path)

        if img_id == "" or study_id == "" or phase_id == "":
            return False
        return True

    def get_umie_img_path(self, img_path: str) -> str:
        """Create a unique path for the image.

        Args:
            img_path (str): Path to the image.

        Returns:
            str: Unique path for the image.
        """
        umie_id = self.get_umie_id(img_path)

        phase_id = self.phase_id_extractor(img_path)

        if phase_id not in self.phases.keys():
            raise ValueError(f"Phase id {phase_id} not in the list of phases.")
        phase_name = self.phases[phase_id]

        new_path = os.path.join(
            self.target_path,
            f"{self.dataset_uid}_{self.dataset_name}",
            phase_name,
            self.image_folder_name,
            umie_id,
        )
        return new_path

    def get_umie_mask_path(self, img_path: str) -> str:
        """Create a unique path for the mask.

        Args:
            img_path (str): Path to the image.

        Returns:
            str: Unique path for the mask.
        """
        umie_id = os.path.basename(img_path)
        phase_name = self.decode_umie_img_path(img_path)[0]

        new_path = os.path.join(
            self.target_path,
            f"{self.dataset_uid}_{self.dataset_name}",
            phase_name,
            self.mask_folder_name,
            umie_id,
        )
        return new_path

    def decode_umie_img_path(self, umie_path: str) -> tuple:
        """Decode the unique image path.

        Args:
            umie_path (str): Unique image path.

        Returns:
            tuple: Tuple containing the phase name, study id, and image id.
        """
        umie_id = os.path.basename(umie_path)
        umie_path_elements = umie_id.split("_")
        phase_name = self.phases[umie_path_elements[1]]
        study_id = umie_path_elements[2]
        img_id = umie_path_elements[3]
        return phase_name, study_id, img_id

    def get_path_without_target_path(self, path: str) -> str:
        """Get the path without the target path.

        Args:
            path (str): Path to the image.

        Returns:
            str: Path to the image without the target path.
        """
        return os.path.relpath(path, self.target_path)
