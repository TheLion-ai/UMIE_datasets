# **Data preparation instruction**
For dicom datasets you will need **pydicom** and **gdcmconv** libraries that are only available in
**Anaconda**.

## 1. CT
* **CT-ORG**
  1. Download CT-ORG dataset from cancerImagingarchive
  2. Use [convertCT_ORG.py](./Train_dataset/CT/convertCT_ORG.py), insert path to your dataset in path (line 11)

* **MosMedData**
  1. Download Chest CT Scans with COVID-19 dataset from mosmed.ai
  2.  Use [converterMosMedAiCOVID.py](./Train_dataset/CT/converterMosMedAiCOVID.py) , insert path to your dataset in path (line 11)

* **KITS-19**
  1. Download KITS-19 dataset according to instruction included in Kits repository
  2. Use [converterKITS19.py](./Train_dataset/CT/converterKITS19.py), insert path to your dataset in path (line 11), in (line 89)
add path to kits.json file

* **LIDC-IDRI**
  1. Download LIDC-IDRI dataset from cancerImagingarchive
  2. delete files that don't match the others (XRAYS)
  3. use [retrieve_data_from_xml.py](./Train_dataset/CT/retrieve_data_from_xml.py) to create nodule masks from xmls, add path to directory (line 150)
  4. use [convertLIDC.py](./Train_dataset/CT/convertLIDC.py), insert path to your pictures in path (line 16), and path to masks (line 18)

## 2. MRI
* **QIN-BRAIN-DSC-MRI**
  1. Download QIN-BRAIN-DSC-MRI dataset from cancerImagingarchive
  2. Use [converterQIN_BRAIN_MRI.py](./Train_dataset/MRI/converterQIN_BRAIN_MRI.py), insert path to your dataset in path (line 11)

* **Brain Tumor Classification (MRI)**
  1. Download Brain Tumor Classification (MRI) dataset from kaggle.com
  2.   Use [converterBrain_Tumor_Classification_MRI.py](./Train_dataset/MRI/converterBrain_Tumor_Classification_MRI.py), insert path to your dataset in path (line 6)

* **Brain-Tumor-Progression**
  1. Download Brain-Tumor-Progression dataset from cancerImagingarchive
  2. Use [converterBrainTumorProgression.py](./Train_dataset/MRI/converterBrainTumorProgression.py), insert path to your dataset in path (line 10)

## 3. X-Ray
* **Chest X-ray 14**
  1. Download from https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345
  2. Use [ConvertCSV_ChestX_ray14.py](./Train_dataset/X_Ray/ConvertCSV_ChestX_ray14.py) and [chest_x_ray14_process.py](./Train_dataset/X_Ray/chest_x_ray14_process.py) Processing images:)
  3. open file [ConvertCSV_ChestX_ray14.py](./Train_dataset/X_Ray/ConvertCSV_ChestX_ray14.py) and change ‚data_csv’ value to path to file‚ Data_Entry_2017_v2020.csv’ in downloaded folder and run the script. It will output file labels.csv.
      - open file [chest_x_ray14_process.py](./Train_dataset/X_Ray/chest_x_ray14_process.py)
      - assign path to folder with source images to variable ‚dir_s’
      - assign path to produced labels.csv file with labels to variable ‚dir_labels’
      - assign destination path for result images to variable ‚dir_d’
      - run script and after execution, processed images will be in destination folder For source images in multiple folders, script can be run multiple times with different paths.

* **Xray Lung segmentation from Chest X-Rays**
  1. Download from: https://www.kaggle.com/nikhilpandey360chest-xray-masksand-labels
  2. Change path variables to downloaded dataset in [lung_segmentation_Xray_unifying.py](./Train_dataset/X_Ray/lung_segmentation_Xray_unifying.py) script
      - masks_path = "dataset_lungs/data Lung Segmentation/masks"
      - images_path = "dataset_lungs/data/Lung Segmentation/CXR_png"
  3. Run script. This will result in unified dataset with masks in new folders

* **Xray CoronaHack -Chest X-Ray-Dataset**
  1. Download from: https://www.kaggle.com/praveengovi/coronahack-chestxraydataset?select=Chest_xray_Corona_Metadata.csv
  2. Change path variables to downloaded dataset in [coronahack_process.py](./Train_dataset/X_Ray/coronahack_process.py) script
      - dir.append('first_path_to_dataset_folder')
      - dir.append('second_path_to_dataset_folder')
      - dir_csv = '' # path to .csv file with labels
      - dir_d = '' # path to destination folder
  3. Run Script
