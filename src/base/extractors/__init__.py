"""
This module contains the base extractors for different types of data.

The extractors extract specific information from the data, such as image IDs, labels, phase IDs, and study IDs.
They label extractor is used to extract labels from the data.
The other extractors are used to extract relevant information to form UMIE ID from the data.
The extractors included in this module are:
- BaseImgIdExtractor: Extracts image IDs. (individual img file identifier)
- BaseLabelExtractor: Extracts labels.
- BasePhaseIdExtractor: Extracts phase IDs.
- BaseStudyIdExtractor: Extracts study IDs. Each medical imaging examination contains many imgs, study id identifies all the imgs from the same examination.
"""
from .img_id import BaseImgIdExtractor
from .label import BaseLabelExtractor
from .phase_id import BasePhaseIdExtractor
from .study_id import BaseStudyIdExtractor
