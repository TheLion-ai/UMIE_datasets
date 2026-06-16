"""Merge several single-structure mask PNGs for one image into one multi-class mask."""

import glob
import json
import os
from collections import defaultdict
from typing import Optional

import cv2  # type: ignore[import-untyped]
import numpy as np

from base.step import BaseStep


class MergeMasks(BaseStep):
    """Merge several single-structure mask PNGs for one image into one multi-class mask.

    Each source mask labels a different anatomical structure with its own
    ``config/masks.py`` colour. For every UMIE image that has more than one mask written
    under the dataset's ``Masks`` folders, the masks are combined into a single multi-class
    mask. Pixels that are already non-zero in an earlier mask (overlaps) are resolved
    according to ``self.format_config.overlap_policy`` (``report`` | ``first`` | ``last`` |
    ``priority``). An overlaps summary is always written to
    ``reports_dir()/mask_merge_report.json``. UMIE ids and folder layout are never changed.
    """

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Merge multiple per-structure masks of each image into one multi-class mask.

        Masks are grouped by their output basename (the UMIE id) within each phase's
        ``Masks`` folder. A group with several distinct mask arrays is merged with the
        configured overlap policy and written back over the group's masks. A per-group
        overlap summary is written to ``reports_dir()/mask_merge_report.json``. ``X`` is
        returned unchanged.

        Args:
            X (list): List of paths to the images.
        Returns:
            list: The unchanged list of image paths.
        """
        print("Merging masks...")
        policy = self.format_config.overlap_policy
        priority = self.format_config.merge_priority
        groups = self._group_mask_paths()
        report: dict = {
            "overlap_policy": policy,
            "merge_priority": priority,
            "groups": [],
        }
        for group_key, mask_paths in sorted(groups.items()):
            if len(mask_paths) < 2:
                continue
            arrays = [array for array in (self._read_mask(path) for path in mask_paths) if array is not None]
            if len(arrays) < 2:
                continue
            merged, overlap_count = self._merge(arrays, policy, priority)
            for path in mask_paths:
                cv2.imwrite(path, merged)
            report["groups"].append(
                {
                    "group": group_key,
                    "masks": [self.get_path_without_target_path(path) for path in mask_paths],
                    "overlap_pixel_count": overlap_count,
                }
            )
        self._write_report(report)
        print(f"Mask merge complete: {len(report['groups'])} group(s) merged.")
        return X

    def _group_mask_paths(self) -> dict[str, list[str]]:
        """Group output mask PNG paths by their basename (the UMIE id).

        Returns:
            dict[str, list[str]]: Mapping of mask basename to the sorted paths sharing it.
        """
        # Single-structure masks for the same image share a UMIE-id basename; they live either
        # directly in a phase's Masks folder or in per-structure subfolders beneath it. Match
        # both by globbing every PNG at any depth under a Masks folder.
        pattern = os.path.join(self.dataset_root, f"**/{self.mask_folder_name}/**/*.png")
        groups: dict[str, list[str]] = defaultdict(list)
        for path in glob.glob(pattern, recursive=True):
            groups[os.path.basename(path)].append(path)
        return {key: sorted(value) for key, value in groups.items()}

    def _read_mask(self, path: str) -> Optional[np.ndarray]:
        """Read a single-channel mask PNG into a numpy array, or None if unreadable.

        Args:
            path (str): Path to the mask PNG.
        Returns:
            Optional[np.ndarray]: The mask array, or None when it cannot be read.
        """
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return mask

    @staticmethod
    def _merge(
        mask_arrays: list[np.ndarray],
        policy: str,
        priority: Optional[list],
    ) -> tuple[np.ndarray, int]:
        """Merge mask arrays into one and resolve overlaps per the chosen policy.

        ``report`` and ``first`` both keep the colour written by the earlier mask on an
        overlapping pixel; ``last`` keeps the colour from the later mask; ``priority`` keeps
        the colour appearing earlier in ``priority`` (descending priority). Overlap pixels are
        counted wherever two or more input masks are simultaneously non-zero.

        Args:
            mask_arrays (list[np.ndarray]): Single-structure masks for the same image.
            policy (str): One of ``report`` | ``first`` | ``last`` | ``priority``.
            priority (Optional[list]): Mask colours in descending priority (policy="priority").
        Returns:
            tuple[np.ndarray, int]: The merged mask and the overlapping pixel count.
        """
        first = mask_arrays[0]
        merged = np.zeros_like(first)
        occupied = np.zeros(first.shape[:2], dtype=bool)
        overlap = np.zeros(first.shape[:2], dtype=bool)

        priority_rank = {int(color): rank for rank, color in enumerate(priority or [])}
        # Higher number == better-established colour already placed; default rank for the
        # "priority" policy keeps unknown colours below any listed colour.
        worst_rank = len(priority_rank) + 1
        current_rank = np.full(first.shape[:2], worst_rank, dtype=np.int64)

        for mask in mask_arrays:
            present = mask > 0
            overlap |= occupied & present

            if policy == "last":
                write = present
            elif policy == "priority":
                incoming_rank = np.full(first.shape[:2], worst_rank, dtype=np.int64)
                for color, rank in priority_rank.items():
                    incoming_rank[mask == color] = rank
                # Lower rank value == higher priority; write where empty or strictly better.
                write = present & (~occupied | (incoming_rank < current_rank))
                current_rank = np.where(write, incoming_rank, current_rank)
            else:  # "report" and "first" keep the earlier colour
                write = present & ~occupied

            merged[write] = mask[write]
            occupied |= present

        return merged, int(np.count_nonzero(overlap))

    def _write_report(self, report: dict) -> None:
        """Write the mask-merge overlaps summary to JSON under the reports dir.

        Args:
            report (dict): The accumulated merge report.
        """
        report_path = os.path.join(self.reports_dir(), "mask_merge_report.json")
        with open(report_path, "w") as handle:
            json.dump(report, handle, indent=2)
