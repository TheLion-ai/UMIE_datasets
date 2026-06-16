"""Compute SHA-256 checksums of dataset outputs and write (or verify) a manifest.

This optional, opt-in reproducibility step (Theme H, Task 27) walks every output file
under ``self.dataset_root`` and records its SHA-256 hash. The manifest is a JSON object
mapping each file's posix path (relative to ``dataset_root``) to its hex digest.

Two modes are controlled by :class:`~base.pipeline.ExportConfig`:

* ``verify_manifest is False`` (default): build the manifest and write it to
  ``reports_dir()/manifest.json``.
* ``verify_manifest is True``: load the existing ``manifest.json`` and re-check the
  current outputs against it, writing the differences to
  ``reports_dir()/manifest_verification.json`` with three categories - ``missing``
  (recorded in the manifest but absent on disk), ``changed`` (hash differs), and
  ``extra`` (present on disk but not in the manifest). The manifest itself is left
  untouched in verify mode.

The ``reports/`` folder (where the manifest and verification report live) and the
manifest file are always excluded from hashing, so the manifest never references itself
and analysis reports never invalidate verification. The step reads files only; it never
alters any image or JSONL output and returns ``X`` unchanged.
"""

import hashlib
import json
import os

from base.step import BaseStep
from constants import REPORTS_FOLDER_NAME


class CreateManifest(BaseStep):
    """Write or verify a SHA-256 manifest of every output file under ``dataset_root``."""

    _MANIFEST_NAME = "manifest.json"
    _VERIFICATION_NAME = "manifest_verification.json"
    _CHUNK_SIZE = 1 << 20  # 1 MiB read buffer for streaming large files

    def transform(self, X: list) -> list:
        """Build or verify the output manifest, leaving the outputs and ``X`` unchanged.

        Args:
            X (list): List of paths passed through the pipeline.

        Returns:
            list: The same ``X``, unchanged.
        """
        if self.export_config.verify_manifest:
            self._verify_manifest()
        elif self.export_config.write_manifest:
            self._write_manifest()
        return X

    def _sha256(self, path: str) -> str:
        """Compute the SHA-256 hex digest of a file, streaming it in chunks.

        Args:
            path (str): Absolute path to the file to hash.

        Returns:
            str: The lowercase hex SHA-256 digest of the file's bytes.
        """
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(self._CHUNK_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _build_manifest(self) -> dict[str, str]:
        """Hash every output file under ``dataset_root`` (excluding the reports folder).

        Returns:
            dict[str, str]: Mapping of posix relative path (to ``dataset_root``) to digest.
        """
        manifest: dict[str, str] = {}
        root = self.dataset_root
        for current_dir, dirnames, filenames in os.walk(root):
            # Never descend into the reports/ folder: it holds the manifest and analysis
            # reports, neither of which is part of the reproducible image/JSONL output.
            if os.path.relpath(current_dir, root) == os.curdir:
                dirnames[:] = [d for d in dirnames if d != REPORTS_FOLDER_NAME]
            for filename in filenames:
                abs_path = os.path.join(current_dir, filename)
                rel_path = os.path.relpath(abs_path, root).replace(os.sep, "/")
                manifest[rel_path] = self._sha256(abs_path)
        return dict(sorted(manifest.items()))

    def _manifest_path(self) -> str:
        """Return the absolute path to the manifest file under ``reports_dir()``."""
        return os.path.join(self.reports_dir(), self._MANIFEST_NAME)

    def _write_manifest(self) -> None:
        """Build the manifest and write it as deterministic JSON to ``reports_dir()``."""
        manifest = self._build_manifest()
        with open(self._manifest_path(), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    def _verify_manifest(self) -> None:
        """Re-check current outputs against an existing manifest and write the report.

        Compares the on-disk outputs to the recorded manifest and writes a
        ``manifest_verification.json`` report with ``missing`` / ``changed`` / ``extra``
        lists. The manifest is not rewritten in this mode.
        """
        manifest_path = self._manifest_path()
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"No manifest to verify at {manifest_path}.")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            recorded: dict[str, str] = json.load(handle)

        current = self._build_manifest()

        missing = sorted(rel for rel in recorded if rel not in current)
        extra = sorted(rel for rel in current if rel not in recorded)
        changed = sorted(rel for rel in recorded if rel in current and recorded[rel] != current[rel])

        report = {"missing": missing, "changed": changed, "extra": extra}
        with open(os.path.join(self.reports_dir(), self._VERIFICATION_NAME), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
