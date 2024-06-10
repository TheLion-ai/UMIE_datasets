"""Preprocessing pipeline for Liver and liver tumor dataset."""
import os
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any

import cv2

from config import mask_encodings_config
from src.pipelines.base_pipeline import BasePipeline, DatasetArgs
from src.steps.add_labels import AddLabels
from src.steps.add_new_ids import AddNewIds
from src.steps.combine_multiple_masks import CombineMultipleMasks
from src.steps.copy_masks import CopyMasks
from src.steps.create_file_tree import CreateFileTree
from src.steps.delete_imgs_with_empty_masks import DeleteImgsWithEmptyMasks
from src.steps.get_file_paths import GetFilePaths


@dataclass
class LITSPipeline(BasePipeline):
    """Preprocessing pipeline for Liver and liver tumor dataset."""

    name: str = field(default="Liver_And_Liver_Tumor")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("get_file_paths", GetFilePaths),
            ("create_file_tree", CreateFileTree),
            ("combine_multiple_masks", CombineMultipleMasks),
            ("copy_masks", CopyMasks),
            ("add_new_ids", AddNewIds),
            ("add_labels", AddLabels),
            # Recommended to delete images without masks, because they contain neither liver nor tumor
            ("delete_imgs_with_empty_masks", DeleteImgsWithEmptyMasks),
        ]
    )

    dataset_args: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(
            img_prefix="volume",  # prefix of the source image file names
            mask_selector="segmentation",
            multiple_masks_selector={"livermask": "liver", "lesionmask": "liver_tumor"},
            phase_extractor=lambda x: "0",
        )
    )

    def img_id_extractor(self, img_path: str) -> str:
        """Retrieve image id from path."""
        basename = os.path.basename(img_path).split(".")[0]
        img_id = basename.rsplit("_", 1)[1]
        return img_id

    def study_id_extractor(self, img_path: str) -> str:
        """Retrieve study id from path."""
        basename = os.path.basename(img_path).split(".")[0]
        study_id = basename.rsplit("_", 1)[0].rsplit("-", 1)[1]
        return study_id

    def get_label(self, img_path: str) -> list:
        """Get image label based on path."""
        mask_path = img_path.replace(self.dataset_args.image_folder_name, self.dataset_args.mask_folder_name)
        mask = cv2.imread(mask_path)
        if mask_encodings_config.mask_encodings["liver_tumor"] in mask:
            return ["Tumor"]
        else:
            return ["good"]

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.dataset_args.img_id_extractor = lambda x: self.img_id_extractor(x)
        self.dataset_args.study_id_extractor = lambda x: self.study_id_extractor(x)

        # Add get_label function to the dataset_args
        self.dataset_args.get_label = partial(self.get_label)
        # Update args with dataset_args
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
