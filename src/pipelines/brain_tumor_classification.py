"""Preprocessing pipeline for brain tumor classification dataset."""


import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.pipelines.base_pipeline import BasePipeline, DatasetArgs
from src.steps.add_labels import AddLabels
from src.steps.add_new_ids import AddNewIds
from src.steps.convert_jpg2png import ConvertJpg2Png
from src.steps.create_file_tree import CreateFileTree
from src.steps.delete_temp_png import DeleteTempPng
from src.steps.get_file_paths import GetFilePaths
from src.steps.get_source_paths import GetSourcePaths


@dataclass
class BrainTumorClassificationPipeline(BasePipeline):
    """Preprocessing pipeline for Brain Tumor Classification dataset."""

    name: str = field(default="Brain_Tumor_Classification_MRI")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
            ("get_file_paths", GetFilePaths),
            ("get_source_paths", GetSourcePaths),
            ("convert_jpg2png", ConvertJpg2Png),
            ("add_new_ids", AddNewIds),
            ("add_labels", AddLabels),
            ("delete_temp_png", DeleteTempPng),
        ]
    )

    dataset_args: DatasetArgs = field(
        default_factory=lambda: DatasetArgs(
            image_folder_name="Images",
            mask_folder_name=None,
            img_prefix="",
        )
    )

    def img_id_extractor(self, img_path: str) -> str:
        """Img id always 0 in this dataset."""
        return "0"

    def study_id_extractor(self, img_path: str) -> str:
        """Extract study id from img path. Img names are not unique.

        To make them unique there is system based on image folder path
        /Training/ -> 0 added /Testing/ -> 1 added then there are 4 folders one for each label.
        This fact is used to assign unique ids
        """
        unique_id_conversion_dict = {
            "glioma_tumor": "00",
            "meningioma_tumor": "01",
            "pituitary_tumor": "10",
            "no_tumor": "11",
        }
        unique_id = ""
        if "Training" in img_path:
            unique_id = "0"
        else:
            unique_id = "1"

        image_folder = os.path.basename(os.path.dirname(img_path))
        unique_id = unique_id + unique_id_conversion_dict[image_folder]

        parent_directory = Path(img_path).parent
        files_in_parent_directory = os.listdir(parent_directory)

        # after conversion to png there are additional png files
        jpg_files = [Path(file).stem for file in files_in_parent_directory if file.lower().endswith(".jpg")]
        # makes sure that we get the same order
        jpg_files.sort()

        img_filename = Path(img_path).name
        return unique_id + str(jpg_files.index(Path(img_filename).stem))

    def get_label(self, img_path: str) -> list:
        """Get label for file. Label is a name of folder in source directory."""
        image_folder = os.path.basename(os.path.dirname(img_path))
        standard_ontology = {
            "glioma_tumor": "Glioma",
            "meningioma_tumor": "Meningioma",
            "pituitary_tumor": "Pituitary",
            "no_tumor": "good",
        }
        return [standard_ontology[image_folder]]

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.dataset_args.img_id_extractor = lambda x: self.img_id_extractor(x)
        self.dataset_args.study_id_extractor = lambda x: self.study_id_extractor(x)
        self.dataset_args.get_label = lambda x: self.get_label(x)

        self.args["compress_after_folder"] = "archive"
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.dataset_args))
