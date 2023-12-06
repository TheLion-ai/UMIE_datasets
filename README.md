# UMIE_datasets

This repository presents a suite of unified scripts to standardize, preprocess, and integrate 882,774 images from 20 open-source medical imaging datasets, spanning modalities such as X-ray, CT, and MR. The scripts allow for seamless and fast download of a diverse medical data set. We create a unified set of annotations allowing for merging the datasets together without mislabelling.


![Preprocessing_modules](dataset_modules.png)

## Datasets
| Dataset | Modality | TASK |
| :------:| :--------:|:-------:|
| [KITS-19](https://github.com/neheller/kits19)   | CT | classification/segmentation|
| [MosMedData](https://mosmed.ai/en/) | CT | classification/segmentation|
| [LIDC-IRI](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254) | CT | segmentation|
| [CT-ORG](https://wiki.cancerimagingarchive.net/display/Public/CT-ORG%3A+CT+volumes+with+multiple+organ+segmentations) | CT | segmentation|
| [Chest X-ray](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345) | XRAY | classification|
| [CoronaHack](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset?select=Chest_xray_Corona_Metadata.csv) | XRAY | classification|
| [Lung segmentation from Chest -Xray](https://www.kaggle.com/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset) | XRAY | classification/segmentation|
| [Brain Tumor Classification](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri) | MRI | classification|
| [Brain Tumor Progression](https://wiki.cancerimagingarchive.net/display/Public/Brain-Tumor-Progression#339481190e2ccc0d07d7455ab87b3ebb625adf48) | MRI | segmentation|
| [QIN-BRAIN-DSC-MRI](https://wiki.cancerimagingarchive.net/display/Public/QIN-BRAIN-DSC-MRI#21267383b89ada0490e14a66b34d72fe8d9d8a0b) | MRI | segmentation|
| [COVID-19 Detection X-Ray](https://www.kaggle.com/darshan1504/covid19-detection-xray-dataset) | XRAY | classification |
| [Shenzhen Hospital Chest X-ray Set](https://www.kaggle.com/yoctoman/shcxr-lung-mask)| X-Ray | Segmentation |
| [Knee Osteoarthritis Dataset with Severity Grading](https://www.kaggle.com/shashwatwork/knee-osteoarthritis-dataset-with-severity)	|	X-Ray | Classification |
| [Finding and Measuring Lungs in CT Data](https://www.kaggle.com/kmader/finding-lungs-in-ct-data?select=lung_stats.csv)	| CT | Segmentation |
| [Brain CT Images with Intracranial Hemorrhage Masks](https://www.kaggle.com/vbookshelf/computed-tomography-ct-images)	| CT | Classification |
| [Liver and Liver Tumor Segmentation](https://www.kaggle.com/andrewmvd/lits-png?select=lits_test.csv)| CT | Classification, Segmentation |
| [Alzheimers Dataset](https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images)	| MRI | Classification |
| [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/jjprotube/brain-mri-images-for-brain-tumor-detection) | MRI | Classification |
| [BrainMetShare](https://aimi.stanford.edu/brainmetshare) | MRI | Segmentation |
| [COCA- Coronary Calcium and chest CTs](https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa) | CT | Segmentation |



# **Data preparation instruction**
## Installing requirements
```bash
poetry install
```
## Creating the dataset
Unfortunately due to the copyright restrictions of the source datasets, we can't share the files directly. To obtain the full dataset you have to download the source datasets yourself and run the preprocessing scripts.

## 1. CT
* **CT-ORG**
  1. Download the [CT-ORG](https://wiki.cancerimagingarchive.net/display/Public/CT-ORG%3A+CT+volumes+with+multiple+organ+segmentations) from the cancer imaging archive
  2. Use [convertCT_ORG.py](./pipelines/CT/convertCT_ORG.py), specify source and target paths.

* **MosMedData**
  1. Download Chest CT Scans with COVID-19 dataset from [MosMedData](https://mosmed.ai/en/)
  2.  Use [converterMosMedAiCOVID.py](./pipelines/CT/converterMosMedAiCOVID.py) , specify source and target paths.

* **KITS-19 or KITS-21**
  1. Download the KITS-19 dataset according to the instructions included in the Kits repository
  2. Use [converterKITS19.py](./pipelines/CT/converterKITS19.py), insert the path to your dataset add the path to the kits.json file.
  3. You can also use these scripts for the KITS-21 dataset but not for KITS-23. KITS-23 has not released the labels for kidney tumor labels.

* **LIDC-IDRI**
  1. Download [LIDC-IDRI](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254)dataset from the cancer imaging archive
  2. delete files that don't match the others (XRAYS)
  3. use [retrieve_data_from_xml.py](./pipelines/CT/retrieve_data_from_xml.py) to create nodule masks from xmls, add path to directory (line 150)
  4. use [convertLIDC.py](./pipelines/CT/convertLIDC.py), specify source and target paths.

* **Finding and Measuring Lungs in CT Data**
  1.  Download the dataset from [Finding and Measuring Lungs in CT Data](https://www.kaggle.com/kmader/finding-lungs-in-ct-data?select=lung_stats.csv).
  2.  Use [test_ct_lungs.py](./pipelines/CT/test_ct_lungs.py) to transform the dataset. Set source paths for images and masks, as well as the target paths.

* **Brain CT Images with Intracranial Hemorrhage Masks**
  1. Download the data from [Finding and Measuring Lungs in CT Data](https://www.kaggle.com/kmader/finding-lungs-in-ct-data?select=lung_stats.csv).
  2. Use [test_ct_brain.py](./pipelines/CT/test_ct_brain.py) to transform the dataset. Specify paths to source images and masks, target images and masks, and a csv file with annotations.

* **Liver and Liver Tumor Segmentation**
  1. Download the dataset from  [Liver and Liver Tumor Segmentation](https://www.kaggle.com/andrewmvd/lits-png?select=lits_test.csv).
  2. Transform the data with [test_ct_liver.py](./pipelines/CT/test_ct_liver.py).

* **COCA- Coronary Calcium and chest CTs**
  1. Download the dataset from [COCA- Coronary Calcium and chest CTs](https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa). You will need to create an account.
  2. Use [coca.py](./pipelines/CT/coca.py) to transform the dataset.


## 2. MRI
* **QIN-BRAIN-DSC-MRI**
  1. Download the [QIN-BRAIN-DSC-MRI](https://wiki.cancerimagingarchive.net/display/Public/QIN-BRAIN-DSC-MRI#21267383b89ada0490e14a66b34d72fe8d9d8a0b) dataset from cancer imaging archive
  2. Use [converterQIN_BRAIN_MRI.py](./pipelines/MRI/converterQIN_BRAIN_MRI.py), specify source and target paths.

* **Brain Tumor Classification (MRI)**
  1. Download the [Brain Tumor Progression](https://wiki.cancerimagingarchive.net/display/Public/Brain-Tumor-Progression#339481190e2ccc0d07d7455ab87b3ebb625adf48) dataset from kaggle.com
  2.   Use [converterBrain_Tumor_Classification_MRI.py](./pipelines/MRI/converterBrain_Tumor_Classification_MRI.py), specify source and target paths.

* **Brain-Tumor-Progression**
  1. Download [Brain Tumor Progression](https://wiki.cancerimagingarchive.net/display/Public/Brain-Tumor-Progression#339481190e2ccc0d07d7455ab87b3ebb625adf48)  dataset from the cancer imaging archive
  2. Use [converterBrainTumorProgression.py](./pipelines/MRI/converterBrainTumorProgression.py), Specify source and target paths.

* **Alzheimers Dataset**
  1. Download the dataset from [Alzheimers Dataset](https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images).
  2. Use [test_mri_alzheimer.py](./pipelines/MRI/test_mri_alzheimer.py) to transform the data. Specify source and target directory.

* **Brain MRI Images for Brain Tumor Detection**
  1. Download the dataset from [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/jjprotube/brain-mri-images-for-brain-tumor-detection).
  2. Use [test_brain_mri.py](./pipelines/MRI/test_brain_mri.py) to transform the datasets.

*  **BrainMetShare**
  1. Download the data from [BrainMetShare](https://aimi.stanford.edu/brainmetshare). You will need to create an account.
  2. Use [brain_met_share](./pipelines/MRI/brain_met_share.py) to transform the dataset.


## 3. X-Ray
* **Chest X-ray 14**
  1. Download from https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345
  2. Use [ConvertCSV_ChestX_ray14.py](./pipelines/X-Ray/ConvertCSV_ChestX_ray14.py) and [chest_x_ray14_process.py](./pipelines/X_Ray/chest_x_ray14_process.py) Processing images:)
  3. open file [ConvertCSV_ChestX_ray14.py](./pipelines/X-Ray/ConvertCSV_ChestX_ray14.py) and change ‚data_csv’ value to path to file‚ Data_Entry_2017_v2020.csv’ in the downloaded folder and run the script. It will output file labels.csv.
      - open file [chest_x_ray14_process.py](./pipelines/X-Ray/chest_x_ray14_process.py)
      - assign the path to the folder with source images to variable ‚dir_s’
      - assign the path to produced labels.csv file with labels to variable ‚dir_labels’
      - assign destination path for result images to variable ‚dir_d’
      - run the script and after execution, processed images will be in the destination folder For source images in multiple folders, the script can be run multiple times with different paths.

* **Xray Lung segmentation from Chest X-Rays**
  1. Download from: [Lung segmentation from Chest -Xray](https://www.kaggle.com/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset)
  2. Specify source and target paths in [lung_segmentation_Xray_unifying.py](./pipelines/X-Ray/lung_segmentation_Xray_unifying.py) script
      - masks_path = "dataset_lungs/data Lung Segmentation/masks"
      - images_path = "dataset_lungs/data/Lung Segmentation/CXR_png"
  3. Run the script. This will result in a unified dataset with masks in new folders

* **Xray CoronaHack -Chest X-Ray-Dataset**
  1. Download from: [CoronaHack](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset?select=Chest_xray_Corona_Metadata.csv)
  2. Change path variables to the downloaded dataset in [coronahack_process.py](./pipelines/X-Ray/coronahack_process.py) script
      - dir.append('first_path_to_dataset_folder')
      - dir.append('second_path_to_dataset_folder')
      - dir_csv = '' # path to .csv file with labels
      - dir_d = '' # path to destination folder
  3. Run Script

* **COVID-19 Detection X-Ray**
  1. Download the dataset from [COVID-19 Detection X-Ray](https://www.kaggle.com/darshan1504/covid19-detection-xray-dataset)
  2. Use the script [covid-19_detection_preprocessing.py](./pipelines/X-Ray/covid-19_detection_preprocessing.py) to transform the files. Use dir_1 and dir_d1 parameters to specify source and target paths.

* **Shenzhen Hospital Chest X-ray Set**
  1. Download the dataset from [Shenzhen Hospital Chest X-ray Set](https://www.kaggle.com/yoctoman/shcxr-lung-mask).
  2. Use the script [test_lung_segmentation.py](./pipelines/X-ray/test_lungs_segmentation.py) to transform the dataset. Set path to source masks and source images, and target images and masks in the marked area.

* **Knee Osteoarthritis Dataset with Severity Grading**
    1. Download the dataset from  [Knee Osteoarthritis Dataset with Severity Grading](https://www.kaggle.com/shashwatwork/knee-osteoarthritis-dataset-with-severity).
    2. Use [knee_images](./pipelines/X-ray/knee_images.py) to transform the dataset. Specify source and target paths.

To preprocess the dataset that is not among the above, search the preprocessing folder. It contains the reusable steps for changing imaging formats, extracting masks, creating file trees, etc. Go to the config file to check which masks and label encodings are available. Append new labels and mask encodings if needed.

Overall the dataset should have ** 882,774** images in **.png** format
* **CT - 500k+**
* **X-Ray - 250k+**
* **MRI - 100k+**
