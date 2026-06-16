"""Detect duplicate and near-duplicate output images using a perceptual (dHash) hash."""

import csv
import glob
import json
import os
from typing import Optional

import cv2  # type: ignore[import-untyped]
import jsonlines
import numpy as np

from base.step import BaseStep
from constants import OutputMode


class DetectDuplicates(BaseStep):
    """Detect duplicate and near-duplicate output images using a perceptual (dHash) hash."""

    def transform(
        self,
        X: list,  # img_paths
    ) -> list:
        """Detect duplicate / near-duplicate output images and write reports; never deletes.

        Computes a dHash for every output image, clusters images whose pairwise Hamming
        distance is within ``quality.duplicate_threshold`` (optionally comparing against an
        external reference-hash JSON), writes a JSON and CSV report and, when
        ``quality.flag_duplicates_in_jsonl`` is set, adds an additive ``duplicate_group_id``
        field to clustered JSONL records. ``X`` is returned unchanged so the step chains.

        Args:
            X (list): List of paths to the images.
        Returns:
            list: The unchanged list of image paths.
        """
        print("Detecting duplicate images...")
        hashes = self._compute_hashes()
        reference = self._load_reference_hashes()
        clusters = self._cluster(hashes, reference)
        self._write_reports(clusters)
        if self.quality.flag_duplicates_in_jsonl:
            self._flag_in_jsonl(clusters)
        print(f"Duplicate detection complete: {len(clusters)} cluster(s) found.")
        return X

    def _output_image_paths(self) -> list:
        """Glob the dataset's output images (PNG in 2D mode, ``.nii.gz`` in 3D mode)."""
        extension = "nii.gz" if self.output_mode == OutputMode.VOLUMES_3D else "png"
        pattern = os.path.join(self.dataset_root, f"**/{self.image_folder_name}/*.{extension}")
        return sorted(glob.glob(pattern, recursive=True))

    def _dhash(self, image: np.ndarray) -> int:
        """Compute the difference hash of a grayscale image as a ``hash_size**2`` bit integer.

        Args:
            image (np.ndarray): Grayscale image array.
        Returns:
            int: The perceptual hash encoded as an integer.
        """
        hash_size = self.quality.duplicate_hash_size
        resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        value = 0
        for bit in diff.flatten():
            value = (value << 1) | int(bool(bit))
        return value

    def _load_image_grayscale(self, path: str) -> Optional[np.ndarray]:
        """Load an output image as a 2D grayscale array, or None if it cannot be read.

        Args:
            path (str): Path to the output image or volume.
        Returns:
            Optional[np.ndarray]: Grayscale array, or None when unreadable.
        """
        if path.endswith(".nii.gz"):
            try:
                import nibabel as nib  # type: ignore[import-untyped]

                data = np.asarray(nib.load(path).get_fdata())  # type: ignore[attr-defined]
                if data.ndim > 2:
                    data = data.reshape(data.shape[0], -1)
                return data.astype(np.float32)
            except Exception:
                return None
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return image

    def _compute_hashes(self) -> dict:
        """Compute the dHash of every readable output image keyed by its UMIE-relative path.

        Returns:
            dict: Mapping of ``umie_path`` to its integer perceptual hash.
        """
        hashes: dict = {}
        for path in self._output_image_paths():
            image = self._load_image_grayscale(path)
            if image is None:
                continue
            umie_path = self.get_path_without_target_path(path)
            hashes[umie_path] = self._dhash(image)
        return hashes

    def _load_reference_hashes(self) -> dict:
        """Load the optional cross-dataset reference-hash JSON ``{umie_path: hash}``.

        Returns:
            dict: Reference hashes, or an empty dict when none are configured.
        """
        ref_path = self.quality.duplicate_reference_hashes
        if not ref_path or not os.path.exists(ref_path):
            return {}
        with open(ref_path) as handle:
            raw = json.load(handle)
        return {key: int(value) for key, value in raw.items()}

    @staticmethod
    def _hamming(left: int, right: int) -> int:
        """Return the Hamming distance between two integer hashes.

        Args:
            left (int): First hash.
            right (int): Second hash.
        Returns:
            int: Number of differing bits.
        """
        return int(bin(left ^ right).count("1"))

    def _cluster(self, hashes: dict, reference: dict) -> list:
        """Cluster images whose pairwise Hamming distance is within the configured threshold.

        Reference hashes participate in clustering so cross-dataset overlaps surface, but a
        cluster is only reported when it contains at least one image from this dataset.

        Args:
            hashes (dict): This dataset's ``{umie_path: hash}`` mapping.
            reference (dict): Optional external ``{umie_path: hash}`` mapping.
        Returns:
            list: Clusters as dicts with ``members``, ``representative`` and ``max_distance``.
        """
        threshold = self.quality.duplicate_threshold
        own_keys = set(hashes.keys())
        combined = {**reference, **hashes}  # this dataset's hashes win on key collisions
        items = list(combined.items())

        parent = {key: key for key, _ in items}

        def find(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: str, right: str) -> None:
            parent[find(left)] = find(right)

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if self._hamming(items[i][1], items[j][1]) <= threshold:
                    union(items[i][0], items[j][0])

        groups: dict = {}
        for key, _ in items:
            groups.setdefault(find(key), []).append(key)

        clusters = []
        for members in groups.values():
            if len(members) < 2:
                continue
            if not any(member in own_keys for member in members):
                continue
            ordered = sorted(members)
            max_distance = 0
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    max_distance = max(max_distance, self._hamming(combined[ordered[i]], combined[ordered[j]]))
            representative = next((member for member in ordered if member in own_keys), ordered[0])
            clusters.append(
                {
                    "members": ordered,
                    "representative": representative,
                    "max_distance": max_distance,
                }
            )
        return sorted(clusters, key=lambda cluster: cluster["representative"])

    def _write_reports(self, clusters: list) -> None:
        """Write the duplicate clusters to a JSON and a CSV report under the reports dir.

        Args:
            clusters (list): The detected duplicate clusters.
        """
        reports_dir = self.reports_dir()
        json_path = os.path.join(reports_dir, "duplicates_report.json")
        with open(json_path, "w") as handle:
            json.dump({"threshold": self.quality.duplicate_threshold, "clusters": clusters}, handle, indent=2)

        csv_path = os.path.join(reports_dir, "duplicates_report.csv")
        with open(csv_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["group_id", "member", "representative", "max_distance"])
            for group_id, cluster in enumerate(clusters):
                for member in cluster["members"]:
                    writer.writerow([group_id, member, cluster["representative"], cluster["max_distance"]])

    def _flag_in_jsonl(self, clusters: list) -> None:
        """Add an additive ``duplicate_group_id`` to clustered records, preserving order.

        Args:
            clusters (list): The detected duplicate clusters.
        """
        if not os.path.exists(self.json_path):
            return
        member_to_group = {}
        for group_id, cluster in enumerate(clusters):
            for member in cluster["members"]:
                member_to_group[member] = group_id

        updated = []
        with jsonlines.open(self.json_path, mode="r") as reader:
            for obj in reader:
                member_group = member_to_group.get(obj.get("umie_path"))
                if member_group is not None:
                    obj["duplicate_group_id"] = member_group
                updated.append(obj)

        with jsonlines.open(self.json_path, mode="w") as writer:
            for obj in updated:
                writer.write(obj)
