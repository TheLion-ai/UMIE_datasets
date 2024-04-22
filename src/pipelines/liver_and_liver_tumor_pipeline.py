"""Preprocessing pipeline for Liver and liver tumor dataset."""
import os
import re
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any

from config import dataset_masks_config
from src.constants import MASK_FOLDER_NAME
from src.pipelines.base_pipeline import BasePipeline, DatasetArgs
from src.steps.add_labels import AddLabels
from src.steps.add_new_ids import AddNewIds
from src.steps.combine_multiple_masks import CombineMultipleMasks
from src.steps.convert_jpg2png import ConvertJpg2Png
from src.steps.copy_masks import CopyMasks
from src.steps.create_blank_masks import CreateBlankMasks
from src.steps.create_file_tree import CreateFileTree
from src.steps.delete_temp_files import DeleteTempFiles
from src.steps.delete_temp_png import DeleteTempPng
from src.steps.get_file_paths import GetFilePaths
from src.steps.masks_to_binary_colors import MasksToBinaryColors
from src.steps.recolor_masks import RecolorMasks


@dataclass
class LiverAndLiverTumorPipeline(BasePipeline):
    """Preprocessing pipeline for Liver and liver tumor dataset."""

    name: str = field(default="Liver_And_Liver_Tumor")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("get_file_paths", GetFilePaths),
            ("create_file_tree", CreateFileTree),
            ("copy_masks", CopyMasks),
            # ("copy_masks", CopyMasks),
            ("combine_multiple_masks", CombineMultipleMasks),
            ("add_new_ids", AddNewIds),
            ("add_labels", AddLabels),
            # Choose either to create blank masks or delete images without masks
            # Recommended to create blank masks because only about 10% images have masks.
            ("create_blank_masks", CreateBlankMasks),
            # ("delete_imgs_without_masks", DeleteImgsWithoutMasks),
            ("delete_temp_files", DeleteTempFiles),
            ("delete_temp_png", DeleteTempPng),
        ]
    )

    dataset_args: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(
            img_prefix="volume",  # prefix of the source image file names
            mask_selector="mask",
            multiple_masks_selector={"livermask": "liver", "lesionmask": "liver_tumor"},
        )
    )

    def img_id_extractor(self, img_path: str) -> str:
        """Retrieve image id from path."""
        return os.path.basename(img_path)

    def study_id_extractor(self, img_path: str) -> str:
        """Retrieve image id from path."""
        # Study id is name of parent folder of images in source directory
        if self.path_args["source_path"] in img_path:
            return os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        if self.path_args["target_path"] in img_path:
            return os.path.basename(img_path)[:3]
        return ""

    def phase_extractor(self, img_path: str) -> str:
        """Retrieve image phase from path."""
        # Phase is the name of folder 2 levels above image in source directory
        if self.path_args["target_path"] in img_path:
            phase_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        else:
            phase_name = os.path.basename(os.path.dirname(img_path))
        lowercase_phases = [x.lower() for x in list(self.args["phases"].values())]
        return list(self.args["phases"].keys())[lowercase_phases.index(phase_name.lower())]

    def get_label(self, img_path: str) -> list:
        """Get image label based on path."""
        # If there is a mask associated with the image in a source directory, then the label is 'hemorrhage'
        if self.path_args["target_path"] in img_path:
            mask_path = img_path.replace(self.dataset_args.image_folder_name, self.dataset_args.mask_folder_name)
            if os.path.exists(mask_path):
                return ["hemorrhage"]
            else:
                return ["good"]
        else:
            # incorrect images directory for invoking get_label
            return []

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.dataset_args.img_id_extractor = lambda x: self.img_id_extractor(x)
        self.dataset_args.study_id_extractor = lambda x: self.study_id_extractor(x)
        self.dataset_args.phase_extractor = self.phase_extractor

        # Add get_label function to the dataset_args
        self.dataset_args.get_label = partial(self.get_label)
        # Update args with dataset_args
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
