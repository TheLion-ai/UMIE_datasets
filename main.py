"""Main script to run dataset preprocessing pipelines."""
# from src.preprocesing.ct import *
import json
import os
import re

import cv2
import numpy as np
import yaml

from config.runner_config import datasets

# datasets = yaml.load(open("config/runner_config.yaml"), Loader=yaml.FullLoader)


for dataset in datasets:
    # Check whether source_path is defined
    if not dataset.args["source_path"]:
        print(f"{dataset} skipped, as no source path found.")
    else:
        print(dataset)
        pipeline = dataset.pipeline
        pipeline.transform(dataset.args["source_path"])
        print(f"{dataset} done.")
