"""
This module contains the base extractors for different types of data.

The extractors included in this module are:
- BaseImgIdExtractor: Extracts image IDs.
- BaseLabelExtractor: Extracts labels.
- BasePhaseIdExtractor: Extracts phase IDs.
- BaseStudyIdExtractor: Extracts study IDs.
"""
from .img_id import BaseImgIdExtractor
from .label import BaseLabelExtractor
from .phase_id import BasePhaseIdExtractor
from .study_id import BaseStudyIdExtractor
