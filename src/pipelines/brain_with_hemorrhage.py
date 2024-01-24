"""Preprocessing pipeline for Brain with hemorrhage dataset."""
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
from src.steps.convert_jpg2png import ConvertJpg2Png
from src.steps.copy_masks import CopyMasks
from src.steps.create_blank_masks import CreateBlankMasks
from src.steps.create_file_tree import CreateFileTree
from src.steps.get_file_paths import GetFilePaths
from src.steps.masks_to_binary_colors import MasksToBinaryColors
from src.steps.recolor_masks import RecolorMasks


@dataclass
class BrainWithHemorrhagePipeline(BasePipeline):
    """Preprocessing pipeline for Brain with hemorrhage dataset."""

    name: str = field(default="Brain_with_hemorrhage")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("get_file_paths", GetFilePaths),
            ("create_file_tree", CreateFileTree),
            ("copy_masks", CopyMasks),
            ("convert_jpg2png", ConvertJpg2Png),
            ("masks_to_binary_colors", MasksToBinaryColors),
            ("recolor_masks", RecolorMasks),
            ("add_new_ids", AddNewIds),
            ("add_labels", AddLabels),
            # Choose either to create blank masks or delete images without masks
            # Recommended to create blank masks because only about 10% images have masks.
            ("create_blank_masks", CreateBlankMasks),
            # ("delete_imgs_without_masks", DeleteImgsWithoutMasks),
        ]
    )

    dataset_args: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(
            img_prefix=".",  # prefix of the source image file names
            segmentation_prefix="_HGE_Seg.",  # prefix of the source mask file names
            mask_selector="_HGE_Seg",
        )
    )

    def img_id_extractor(self, img_path: str) -> str:
        """Retrieve image id from path."""
        if self.path_args["source_path"] in img_path:
            return os.path.basename(os.path.dirname(os.path.dirname(img_path))) + os.path.basename(img_path)
        if self.path_args["target_path"] in img_path:
            return os.path.basename(img_path)[3:]
        return ""

    def study_id_extractor(self, img_path: str) -> str:
        """Retrieve image id from path."""
        if self.path_args["source_path"] in img_path:
            return os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        if self.path_args["target_path"] in img_path:
            return os.path.basename(img_path)[:3]
        return ""

    def phase_extractor(self, img_path: str) -> str:
        """Retrieve image phase from path."""
        if self.path_args["target_path"] in img_path:
            phase_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        else:
            phase_name = os.path.basename(os.path.dirname(img_path))
        lowercase_phases = [x.lower() for x in list(self.args["phases"].values())]
        return list(self.args["phases"].keys())[lowercase_phases.index(phase_name.lower())]

    def get_label(self, img_path: str) -> list:
        """Get image label based on path."""
        if self.path_args["target_path"] in img_path:
            mask_path = img_path.replace(self.dataset_args.image_folder_name, self.dataset_args.mask_folder_name)
            if os.path.exists(mask_path):
                return ["hemorrhage"]
            else:
                return ["good"]
        else:
            dir = os.path.dirname(img_path)
            name = os.path.basename(img_path)
            name, ext = os.path.splitext(name)
            mask_path = os.path.join(dir, name + "_HGE_Seg" + ext)
            if os.path.exists(mask_path):
                return ["hemorrhage"]
            else:
                return ["good"]

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.dataset_args.img_id_extractor = lambda x: self.img_id_extractor(x)
        self.dataset_args.study_id_extractor = lambda x: self.study_id_extractor(x)
        self.dataset_args.phase_extractor = self.phase_extractor

        # Add get_label function to the dataset_args
        self.dataset_args.get_label = partial(self.get_label)
        # Update args with dataset_args
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
