"""Canonical license & source-attribution table for the integrated datasets (Task 23).

This is the single source of truth consumed by:
- the ``AddProvenance`` step (``src/steps/add_provenance.py``), which writes additive
  ``license`` / ``source_dataset`` / ``source_citation`` fields into the JSONL records, and
- the HuggingFace export (``src/steps/export_huggingface.py``) and dataset-card generation,
  which surface the same machine-readable license + attribution.

Keyed by ``DatasetArgs.dataset_name``. ``license`` uses an SPDX-style identifier where one
exists, or a short human-readable string otherwise. Unknown entries fall back to
``DEFAULT_PROVENANCE`` so the field is always populated (and flagged as ``unknown``).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Provenance:
    """License and source attribution for one integrated dataset."""

    license: str  # SPDX identifier or short human-readable license string
    source_dataset: str  # canonical name of the original source dataset
    source_url: Optional[str] = None  # where the source dataset can be obtained
    source_citation: Optional[str] = None  # short citation string to carry through to HF cards / docs
    redistributable: Optional[bool] = None  # whether redistribution is permitted (None = unknown / check license)


DEFAULT_PROVENANCE = Provenance(
    license="unknown",
    source_dataset="unknown",
    redistributable=None,
)

# Best-effort table for the currently-integrated datasets. License strings should be verified
# against each source's terms before redistribution; entries marked "unknown" need confirmation.
PROVENANCE: dict[str, Provenance] = {
    "kits23": Provenance(
        license="CC-BY-NC-SA-4.0",
        source_dataset="KiTS23 (Kidney Tumor Segmentation Challenge 2023)",
        source_url="https://kits-challenge.org/kits23/",
        source_citation="Heller et al., The KiTS21 Challenge, arXiv:2307.01984 (KiTS23).",
        redistributable=True,
    ),
    "coronahack": Provenance(
        license="unknown",
        source_dataset="CoronaHack Chest X-Ray Dataset",
        source_url="https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset",
    ),
    "alzheimers": Provenance(
        license="unknown",
        source_dataset="Alzheimer's MRI Dataset",
        source_url="https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset",
    ),
    "brain_tumor_classification": Provenance(
        license="unknown",
        source_dataset="Brain Tumor Classification (MRI)",
        source_url="https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri",
    ),
    "covid19_detection": Provenance(
        license="unknown",
        source_dataset="COVID-19 Chest X-ray detection",
    ),
    "finding_and_measuring_lungs": Provenance(
        license="unknown",
        source_dataset="Finding and Measuring Lungs in CT Data",
        source_url="https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data",
    ),
    "brain_with_intracranial_hemorrhage": Provenance(
        license="CC-BY-4.0",
        source_dataset="Computed Tomography Images for Intracranial Hemorrhage Detection",
        source_url="https://physionet.org/content/ct-ich/",
    ),
    "lits": Provenance(
        license="CC-BY-NC-ND-4.0",
        source_dataset="LiTS - Liver Tumor Segmentation Challenge",
        redistributable=False,
    ),
    "brain_tumor_detection": Provenance(
        license="unknown",
        source_dataset="Brain MRI Images for Brain Tumor Detection",
    ),
    "knee_osteoarthritis": Provenance(
        license="unknown",
        source_dataset="Knee Osteoarthritis Dataset with Severity Grading",
    ),
    "brain_tumor_progression": Provenance(
        license="CC-BY-3.0",
        source_dataset="Brain-Tumor-Progression (TCIA)",
        source_url="https://www.cancerimagingarchive.net/collection/brain-tumor-progression/",
        redistributable=True,
    ),
    "chest_xray14": Provenance(
        license="No restriction (NIH public)",
        source_dataset="NIH ChestX-ray14",
        source_url="https://nihcc.app.box.com/v/ChestXray-NIHCC",
        source_citation="Wang et al., ChestX-ray8, CVPR 2017.",
        redistributable=True,
    ),
    "coca": Provenance(
        license="Stanford AIMI (research use)",
        source_dataset="COCA - Coronary Calcium and chest CTs",
        source_url="https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa",
    ),
    "brain_met_share": Provenance(
        license="Stanford AIMI (research use)",
        source_dataset="BrainMetShare",
        source_url="https://stanfordaimi.azurewebsites.net/datasets/",
    ),
    "ct_org": Provenance(
        license="CC-BY-3.0",
        source_dataset="CT-ORG: CT volumes with multiple organ segmentations (TCIA)",
        source_url="https://www.cancerimagingarchive.net/collection/ct-org/",
        redistributable=True,
    ),
    "lidc_idri": Provenance(
        license="CC-BY-3.0",
        source_dataset="LIDC-IDRI (TCIA)",
        source_url="https://www.cancerimagingarchive.net/collection/lidc-idri/",
        redistributable=True,
    ),
    "cmmd": Provenance(
        license="CC-BY-4.0",
        source_dataset="CMMD - The Chinese Mammography Database (TCIA)",
        source_url="https://www.cancerimagingarchive.net/collection/cmmd/",
        redistributable=True,
    ),
}


def get_provenance(dataset_name: str) -> Provenance:
    """Return the provenance record for ``dataset_name``, or ``DEFAULT_PROVENANCE`` if unknown."""
    return PROVENANCE.get(dataset_name, DEFAULT_PROVENANCE)
