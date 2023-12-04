"""Config file with the paths to the datasets and their parameters."""
from abc import abstractmethod

from src.pipelines.kits21 import KITS21Pipeline

datasets = [
    KITS21Pipeline(
        path_args={
            "source_path": "/home/basia/kits_dummy",
            "target_path": "./data/",
            "labels_path": "/home/basia/kits_dummy/kits.json",
        },
    )
]


# StanfordCOCA:
#   import_statement: "from src.pipelines.ct.stanford_coca import preprocess_coca"
#   function_name: "preprocess_coca"
#   args:
#     source_path: ""
#     target_path: "./data/"
#     masks_path: ""
# StanfordBrainMET:
#   import_statement: "from src.pipelines.mri.stanford_brain_met import preprocess_stanford_brain_met"
#   function_name: "preprocess_stanford_brain_met"
#   args:
#     source_path: ""
#     target_path: "./data/"
