"""Preprocessing pipeline for Alzheimers dataset."""
import glob
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from config import dataset_config
from src.constants import IMG_FOLDER_NAME
from src.pipelines.base_pipeline import BasePipeline, PipelineArgs
from src.steps.add_labels import AddLabels
from src.steps.add_new_ids import AddNewIds
from src.steps.convert_jpg2png import ConvertJpg2Png
from src.steps.create_file_tree import CreateFileTree
from src.steps.delete_temp_png import DeleteTempPng
from src.steps.get_file_paths import GetFilePaths


@dataclass
class AlzheimersPipeline(BasePipeline):
    """Preprocessing pipeline for Alzheimers dataset."""

    name: str = field(default="alzheimers")  # dataset name used in configs
    steps: list = field(
        default_factory=lambda: [
            ("create_file_tree", CreateFileTree),
            ("get_file_paths", GetFilePaths),
            ("convert_jpg2png", ConvertJpg2Png),
            ("add_new_ids", AddNewIds),
            ("add_labels", AddLabels),
            ("delete_temp_png", DeleteTempPng),
        ]
    )
    dataset_args: dataset_config.alzheimers = field(default_factory=lambda: dataset_config.alzheimers)
    pipeline_args: PipelineArgs = field(
        default_factory=lambda: PipelineArgs(
            phase_extractor=lambda x: "0",  # All images are from the same phase
            image_folder_name=IMG_FOLDER_NAME,
            mask_folder_name=None,
            img_prefix="",
        )
    )

    # Filenames in source directory are not unique, so additional id is added to each study_id, based on parent folder
    # name to make them unique across the whole dataset.

    # Ids added to study id based on the name of parent folder name for a 'test' source directory.
    ids_dict_test = {
        "NonDemented": "0",
        "VeryMildDemented": "1",
        "MildDemented": "2",
        "ModerateDemented": "3",
    }
    # Ids added to study id based on the name of parent folder name for a 'train' source directory.
    ids_dict_train = {
        "nonDem": "000",
        "verymildDem": "111",
        "mildDem": "222",
        "moderateDem": "333",
    }
    inv_ids_dict_train = {v: k for k, v in ids_dict_train.items()}

    def study_id_extractor(self, img_path: str) -> str:
        """Extract study id from img path."""
        basename = os.path.splitext(os.path.basename(img_path))[0]
        # If brackets exist in filename, then study id is within them, else it is 0.
        if "(" in basename:
            pattern = r"[()]"
            study_id = re.split(pattern, basename)[1]
        else:
            study_id = "0"
        # Add identifier based on folder name to make new file name unique across dataset
        if "train" in img_path:
            for id in self.ids_dict_train.keys():
                basename = basename.replace(id, self.ids_dict_train[id])
            study_id = study_id + basename
        else:
            folder = os.path.basename(os.path.dirname(img_path))
            for id in self.ids_dict_test.keys():
                folder = folder.replace(id, self.ids_dict_test[id])
            study_id = folder + study_id
        return study_id

    def img_id_extractor(self, img_path: str) -> str:
        """Retrieve image id from path."""
        img_id = os.path.basename(img_path)
        ext = os.path.splitext(img_id)[1]
        if "train" in img_path:
            img_id = "00"
        else:
            img_id = img_id[:2]
        img_id = img_id + ext
        return img_id

    def reverse_filename(self, img_path: str) -> str:
        """Convert image target name to name in source directory."""
        img_id = os.path.basename(img_path)
        # ext = os.path.splitext(img_id)[1]
        basename = os.path.splitext(img_id)[0]
        study_id = re.split("_", basename)[2]
        img_id = re.split("_", basename)[3]
        filename = ""
        if len(study_id) < 5:
            if study_id[1:] != "0":
                filename = study_id[1:]
                filename = img_id + filename
            else:
                filename = img_id
        else:
            lab = study_id[1:4]
            for id in self.inv_ids_dict_train.keys():
                lab = lab.replace(id, self.inv_ids_dict_train[id])
            filename = lab + study_id[4:]
        filename = filename + ".jpg"
        return filename

    # Changing labels from dataset (folders names) to match standard

    files_source: list[str] = field(default_factory=list)

    def get_label(self, img_path: str) -> list:
        """Get label for file. Label is a name of folder in source directory."""
        # filename_source = os.path.basename(img_path).split("_")[-1].replace(".png", ".jpg")

        filename_source = self.reverse_filename(img_path)
        file_source_path = [path for path in self.files_source if filename_source == os.path.basename(path)][0]
        label = os.path.basename(os.path.dirname(file_source_path))
        radlex_label = self.args["labels"][label]

        return radlex_label

    def prepare_pipeline(self) -> None:
        """Post initialization actions."""
        self.pipeline_args.img_id_extractor = lambda x: self.img_id_extractor(x)
        self.pipeline_args.study_id_extractor = lambda x: self.study_id_extractor(x)
        # List of paths to files in source directory with changed names. It will be later used to get labels.
        self.files_source = glob.glob(os.path.join(self.args["source_path"], "**"), recursive=True)
        self.files_source = [name.replace("(", "").replace(")", "").replace(" ", "") for name in self.files_source]

        self.pipeline_args.get_label = lambda x: self.get_label(x)
        # Add dataset specific arguments to the pipeline arguments
        self.args: dict[str, Any] = dict(**self.args, **asdict(self.pipeline_args))
