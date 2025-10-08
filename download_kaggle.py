import kagglehub
import os

os.environ['KAGGLEHUB_CACHE'] = './' # automatically will put in the /datasets directory
# Download latest version


datasets = [

  {
    "dataset": "praveengovi/coronahack-chest-xraydataset", # 1 Xray CoronaHack - Chest x Ray dataset
    "name": "CORONA_HACK"
  },
  {
    "dataset": "preetpalsingh25/alzheimers-dataset-4-class-of-images", # 2. Alzheimer's Dataset
    "name": "ALZHEIMER"
  },
  {
    "dataset": "masoudnickparvar/brain-tumor-mri-dataset",
    # "dataset": "nikhilpandey360/chest-xray-masks-and-labels", # 3. Brain Tumor Classification (MRI)
    "name": "BRAIN_TUMOR"
  },
  {
    "dataset": "darshan1504/covid19-detection-xray-dataset", # 4 COVID-19 Detection X-Ray
    "name": "COVID19_DETECTION"
  },
  {
    "dataset": "kmader/finding-lungs-in-ct-data", # 5. Finding and Measuring Lungs in CT Data
    "name": "FAM_LUNGS"
  },
  {
    "dataset": "vbookshelf/computed-tomography-ct-images", # 6. Brain CT Images with Intracranial Hemorrhage Masks
    "name": "BRAIN_CT"
  },
  {
    "dataset": "andrewmvd/lits-png", # 7. Liver and Liver Tumor Segmentation (LITS)
    "name": "LITS"
  },
  {
    "dataset": "jjprotube/brain-mri-images-for-brain-tumor-detection", # 8. Brain MRI Images for Brain Tumor Detection
    "name": "BRAIN_MRI"
  }  
  
]

config_file="./.pipeline.env"
with open(config_file, "w") as file:

  for item in datasets:
    path = kagglehub.dataset_download(item["dataset"])
    dataset_path=f"{item["name"]}={path.replace(".", os.getcwd())}\n"
    print(dataset_path)
    file.write(dataset_path)


  file.write(f"KITS23={os.getcwd()}/datasets/kits23/dataset\n")
  
  file.close()