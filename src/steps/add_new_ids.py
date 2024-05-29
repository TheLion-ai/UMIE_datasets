"""Change img ids to match the format of the rest of the dataset."""
import csv
import jsonlines
import glob
import os
import shutil
from typing import Callable

import numpy as np
from sklearn.base import TransformerMixin
from tqdm import tqdm


class AddNewIds(TransformerMixin):
    """Change img ids to match the format of the rest of the dataset."""

    def __init__(
        self,
        target_path: str,
        json_path: str,
        dataset_name: str,
        dataset_uid: str,
        phases: dict,
        image_folder_name: str,
        mask_folder_name: str,
        img_id_extractor: Callable = lambda x: os.path.basename(x),
        study_id_extractor: Callable = lambda x: x,
        phase_extractor: Callable = lambda x: x,
        segmentation_prefix: str = "segmentations",
        **kwargs: dict,
    ):
        """Change img ids to match the format of the rest of the dataset.

        Args:
            target_path (str): Path to the target folder.
            json_path: (str): path to jsonlines with info about individual image in the target dataset,
            dataset_name (str): Name of the dataset.
            dataset_uid (str): Unique identifier of the dataset.
            phases (dict): Dictionary with phases and their names.
            image_folder_name (str): Name of the folder with images. Defaults to "Images".
            mask_folder_name (str): Name of the folder with masks. Defaults to "Masks".
            img_id_extractor (Callable, optional): Function to extract image id from the path. Defaults to lambda x: os.path.basename(x).
            study_id_extractor (Callable, optional): Function to extract study id from the path. Defaults to lambda x: x.
            phase_extractor (Callable, optional): Function to extract phase id from the path. Defaults to lambda x: x.
            segmentation_prefix (str, optional): String to select masks. Defaults to "segmentations".
        """
        self.target_path = target_path
        self.json_path = json_path
        self.dataset_name = dataset_name
        self.dataset_uid = dataset_uid
        self.phases = phases
        self.image_folder_name = image_folder_name
        self.mask_folder_name = mask_folder_name
        self.img_id_extractor = img_id_extractor
        self.study_id_extractor = study_id_extractor
        self.phase_extractor = phase_extractor
        self.segmentation_prefix = segmentation_prefix
        self.paths_data = np.array([])

    def transform(
        self,
        X: list,
    ) -> list:
        """Change img ids to match the format of the rest of the dataset.

        Args:
            X (list): List of paths to the images.
        Returns:
            list: List of paths to the images with labels.
        """
        print("Adding new ids to the dataset...")
        if len(X) == 0:
            raise ValueError("No list of files provided.")
        if os.path.exists(os.path.join(self.target_path, "source_paths.csv")):
            self.paths_data = np.array(list(csv.reader(open(os.path.join(self.target_path, "source_paths.csv")))))

        self.new_json = []
        for img_path in tqdm(X):
            self.add_new_ids(img_path)

        # Update JSON file
        with jsonlines.open(self.json_path, 'w') as writer:
            for obj in self.new_json:
                writer.write(obj)


        if os.path.exists(os.path.join(self.target_path, "source_paths.csv")):
            with open(os.path.join(self.target_path, "source_paths.csv"), "w", newline="") as temp_file:
                writer = csv.writer(temp_file)
                writer.writerows(list(self.paths_data))

        root_path = os.path.join(self.target_path, f"{self.dataset_uid}_{self.dataset_name}")
        new_paths = glob.glob(os.path.join(root_path, f"**/{self.image_folder_name}/*.png"), recursive=True)
        return new_paths
    
    def _update_json(self,
                    new_file_name,
                    phase_name,
                    study_id,
                    has_mask)->None:
        """Update JSON file with the infomration about the images."""
        comparative = ''
        if "PRE" in new_file_name:
            comparative = "PRE"
        elif "POST" in new_file_name:
            comparative = "POST"
        
        img_info = {}
        new_file_name = new_file_name.split(".")[0]
        img_info = {
            'file_name': new_file_name,
            'dataset_name': self.dataset_name,
            'dataset_uid': self.dataset_uid, 
            'phase_name': phase_name,
            'comparative': comparative,
            'study_id': study_id,
            'has_mask': has_mask, 
            'labels': [],
        }

        self.new_json.append(img_info)
        

    def add_new_ids(self, img_path: str) -> None:
        """Change img ids to match the format of the rest of the dataset.

        Args:
            img_path (str): Path to the image.
        """
        # Extract relevant information from the source path
        # The logic of the extraction functions depends on the dataset
        img_id = self.img_id_extractor(img_path)
        study_id = self.study_id_extractor(img_path)
        phase_id = self.phase_extractor(img_path)
        if phase_id not in self.phases.keys():
            raise ValueError(f"Phase {phase_id} not in the phases dictionary.")
        elif self.segmentation_prefix in img_path:
            return None
        elif img_id is None or study_id is None or phase_id is None:
            # Mechanism for skipping images
            return None
        phase_name = self.phases[phase_id]
        new_file_name = f"{self.dataset_uid}_{phase_id}_{study_id}_{img_id}"
        # update file names in temporary csv file data
        temporary_id = f"{phase_id}_{study_id}_{img_id}"
        if len(self.paths_data) > 0 and temporary_id in self.paths_data[:, 0]:
            self.paths_data[:, 0][self.paths_data[:, 0] == temporary_id] = new_file_name
        if ".png" not in new_file_name:
            new_file_name = new_file_name + ".png"
        new_path = os.path.join(
            self.target_path,
            f"{self.dataset_uid}_{self.dataset_name}",
            phase_name,
            self.image_folder_name,
            new_file_name,
        )

        if not os.path.exists(new_path):
            if self.target_path in img_path:
                os.rename(img_path, new_path)
            else:
                shutil.copy2(img_path, new_path)

        has_mask = False
        if self.mask_folder_name is not None:
            mask_path = img_path.replace(self.image_folder_name, self.mask_folder_name)
            new_mask_path = new_path.replace(self.image_folder_name, self.mask_folder_name)
            if os.path.exists(new_mask_path):
                has_mask = True
            if self.mask_folder_name in img_path:
                if os.path.exists(mask_path):
                    os.rename(mask_path, new_mask_path)
                    has_mask = True

        self._update_json(new_file_name, phase_name, study_id, has_mask)