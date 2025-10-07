import kagglehub
import os

os.environ['KAGGLEHUB_CACHE'] = './' # automatically will put in the /datasets directory
# Download latest version

# 1 Xray CoronaHack - Chest x Ray dataset
coronahack_path = kagglehub.dataset_download("praveengovi/coronahack-chest-xraydataset")
print("Path to dataset files:", coronahack_path)


# 2. Alzheimer's Dataset
alzheimers_path = kagglehub.dataset_download("preetpalsingh25/alzheimers-dataset-4-class-of-images")
print("Path to dataset files:", alzheimers_path)


# 3. Brain Tumor Classification (MRI)
brain_tumor_path = kagglehub.dataset_download("nikhilpandey360/chest-xray-masks-and-labels")
print("Path to dataset files:", brain_tumor_path)



# COVID-19 Detection X-Ray
covid19_xray_path = kagglehub.dataset_download("darshan1504/covid19-detection-xray-dataset")
print("Path to dataset files:", covid19_xray_path)



# 5. Finding and Measuring Lungs in CT Data
measuring_lungs_path = kagglehub.dataset_download("kmader/finding-lungs-in-ct-data")
print("Path to dataset files:", measuring_lungs_path)


# 6. Brain CT Images with Intracranial Hemorrhage Masks
ct_path = kagglehub.dataset_download("vbookshelf/computed-tomography-ct-images")
print("Path to dataset files:", ct_path)


# 7. Liver and Liver Tumor Segmentation (LITS)
lits_path = kagglehub.dataset_download("andrewmvd/lits-png")
print("Path to dataset files:", lits_path)

# 8. Brain MRI Images for Brain Tumor Detection
brain_tumor_path = kagglehub.dataset_download("jjprotube/brain-mri-images-for-brain-tumor-detection")
print("Path to dataset files:", brain_tumor_path)