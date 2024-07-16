"""
This module contains the pipeline classes for opensource medical imaging datasets supported by UMIE.

Each pipeline class defines how to preprocess a specific dataset.
A pipline class consists of a set of steps that are executed in a specific order.
The pipelines included in this module are:
- AlzheimersPipeline
- BrainMETSharePipeline
- BrainTumorClassificationPipeline
- BrainTumorDetectionPipeline
- BrainTumorProgressionPipeline
- BrainWithIntracranialHemorrhagePipeline
- ChestXray14Pipeline
- COCAPipeline
- CoronaHackPipeline
- COVID19DetectionPipeline
- FindingAndMeasuringLungsPipeline
- KITS23Pipeline
- KneeOsteoarthritisPipeline
- LITSPipeline
"""
from .alzheimers import AlzheimersPipeline
from .brain_met_share import BrainMETSharePipeline
from .brain_tumor_classification import BrainTumorClassificationPipeline
from .brain_tumor_detection import BrainTumorDetectionPipeline
from .brain_tumor_progression import BrainTumorProgressionPipeline
from .brain_with_intracranial_hemorrhage import BrainWithIntracranialHemorrhagePipeline
from .chest_xray14 import ChestXray14Pipeline
from .coca import COCAPipeline
from .coronahack import CoronaHackPipeline
from .covid19_detection import COVID19DetectionPipeline
from .finding_and_measuring_lungs import FindingAndMeasuringLungsPipeline
from .kits23 import KITS23Pipeline
from .knee_osteoarthritis import KneeOsteoarthritisPipeline
from .lits import LITSPipeline
