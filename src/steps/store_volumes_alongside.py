"""Combined 2D+3D output: keep the volumetric ``.nii.gz`` alongside the 2D PNG slices (Theme M, Task 41).

When a run opts into ``OutputMode.SLICES_2D_AND_VOLUMES_3D`` the normal 2D pipeline still writes one
PNG per slice under ``<modality>/Images/``; this step then copies each **source** image volume into a
sibling ``<modality>/Volumes/`` folder under the same UMIE id, so the study has both representations.
The volume is the source of truth for 3D geometry, and the matching UMIE-id basename (plus the v2
``nifti_files`` block) cross-references the PNG slices to their NIfTI volume.

It is a no-op unless combined mode is selected, and only ever *adds* a ``Volumes/`` folder, so the 2D
PNG output and the existing golden file trees are unchanged.
"""

import glob
import os
import shutil

from base.step import BaseStep
from constants import OutputMode

VOLUMES_FOLDER_NAME = "Volumes"  # sibling of Images/Masks holding the volumetric .nii.gz


class StoreVolumesAlongside(BaseStep):
    """Copy source NIfTI image volumes into ``<modality>/Volumes/`` in combined output mode."""

    def transform(self, X: list) -> list:
        """Place each source image volume alongside the PNG slices (or do nothing outside combined mode).

        Args:
            X (list): Paths flowing through the pipeline (returned unchanged).

        Returns:
            list: ``X`` unchanged - this step only adds the volumetric representation.
        """
        if self.output_mode != OutputMode.SLICES_2D_AND_VOLUMES_3D:
            return X

        for volume_path in glob.glob(os.path.join(self.source_path, "**", "*.nii.gz"), recursive=True):
            if self._is_mask(volume_path):
                continue
            if self.img_selector is not None and not self.img_selector(volume_path):
                continue
            self._store_volume(volume_path)
        return X

    def _is_mask(self, path: str) -> bool:
        """Return True when ``path`` is a segmentation volume (never stored as an image volume)."""
        return bool(self.segmentation_prefix) and self.segmentation_prefix in os.path.basename(path)

    def _store_volume(self, volume_path: str) -> None:
        """Copy one source volume to ``<modality>/Volumes/<umie_volume_id>.nii.gz`` (geometry preserved)."""
        if not self.validate_umie_path(volume_path):
            return
        # Build the UMIE id with the volumetric extension directly (the same id components as the
        # PNG slices, so the two representations share a basename and cross-reference each other).
        img_id = self.img_id_extractor(volume_path)
        for ext in (".nii.gz", ".gz", ".nii", ".png"):
            if img_id.endswith(ext):
                img_id = img_id[: -len(ext)]
                break
        study_id = self.study_id_extractor(volume_path)
        modality_id = self.modality_id_extractor(volume_path)
        umie_volume_id = f"{self.dataset_uid}_{modality_id}_{study_id}_{img_id}.nii.gz"

        modality_name = self.modalities[modality_id]
        volumes_dir = os.path.join(self.dataset_root, modality_name, VOLUMES_FOLDER_NAME)
        os.makedirs(volumes_dir, exist_ok=True)
        dest = os.path.join(volumes_dir, umie_volume_id)
        if not os.path.exists(dest):
            shutil.copy2(volume_path, dest)
