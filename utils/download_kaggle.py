import os

import kagglehub

from src.constants import DATASETS_DOWNLOAD_PATH

os.environ["KAGGLEHUB_CACHE"] = DATASETS_DOWNLOAD_PATH  # automatically will put in the /datasets directory
# Download latest version


datasets = [
    {
        "dataset": "praveengovi/coronahack-chest-xraydataset",  # 1 Xray CoronaHack - Chest x Ray dataset
        "name": "CORONA_HACK",
    },
    {"dataset": "preetpalsingh25/alzheimers-dataset-4-class-of-images", "name": "ALZHEIMER"},  # 2. Alzheimer's Dataset
    {
        "dataset": "masoudnickparvar/brain-tumor-mri-dataset",
        # "dataset": "nikhilpandey360/chest-xray-masks-and-labels", # 3. Brain Tumor Classification (MRI)
        "name": "BRAIN_TUMOR",
    },
    {
        "dataset": "darshan1504/covid19-detection-xray-dataset",  # 4 COVID-19 Detection X-Ray
        "name": "COVID19_DETECTION",
    },
    {"dataset": "kmader/finding-lungs-in-ct-data", "name": "FAM_LUNGS"},  # 5. Finding and Measuring Lungs in CT Data
    {
        "dataset": "vbookshelf/computed-tomography-ct-images",  # 6. Brain CT Images with Intracranial Hemorrhage Masks
        "name": "BRAIN_CT",
    },
    {"dataset": "andrewmvd/lits-png", "name": "LITS"},  # 7. Liver and Liver Tumor Segmentation (LITS)
    {
        "dataset": "jjprotube/brain-mri-images-for-brain-tumor-detection",  # 8. Brain MRI Images for Brain Tumor Detection
        "name": "BRAIN_MRI",
    },
    {
        "dataset": "shashwatwork/knee-osteoarthritis-dataset-with-severity",  # 9. Knee Osteoarthrithis Dataset with Severity Grading
        "name": "KNEE_OSTEO",
    },
]

config_file = "./.pipeline.env"
with open(config_file, "a") as file:

    for item in datasets:
        path = kagglehub.dataset_download(item["dataset"])
        dataset_path = item["name"] + "=" + path.replace(".", os.getcwd()) + "\n"
        print(dataset_path)
        file.write(dataset_path)

    file.close()
