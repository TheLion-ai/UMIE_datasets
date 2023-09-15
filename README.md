# UMIE_datasets

This reposotory presents a suite of unified scripts to standardize, preprocess, and integrate 882,774 images from 20 open-source medical imaging datasets, spanning modalities such as X-ray, CT, and MR. The scripts allow for seamless and fast download of a diverse medical data set. We create a unified set of annotations allowing for merging the datasets together without mislabelling.
## Datasets
| Dataset | Modality | TASK |
| :------:| :--------:|:-------:| 
| [KITS-19](https://github.com/neheller/kits19)   | CT | classification/segmentation|
| [MosMedData](https://mosmed.ai/en/) | CT | classification/segmentation|
| [LIDC-IRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI#) | CT | segmentation|
| [CT-ORG](https://wiki.cancerimagingarchive.net/display/Public/CT-ORG%3A+CT+volumes+with+multiple+organ+segmentations) | CT | segmentation|
| [Chest X-ray](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345) | XRAY | classification|
| [CoronaHack](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset?select=Chest_xray_Corona_Metadata.csv) | XRAY | classification|
| [Lung segmentation from Chest -Xray](https://www.kaggle.com/nikhilpandey360/lung-segmentation-from-chest-x-ray-dataset) | XRAY | classification/segmentation|
| [Brain Tumor Classification](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri) | MRI | classification|
| [Brain Tumor Progression](https://wiki.cancerimagingarchive.net/display/Public/Brain-Tumor-Progression#339481190e2ccc0d07d7455ab87b3ebb625adf48) | MRI | segmentation|
| [QIN-BRAIN-DSC-MRI](https://wiki.cancerimagingarchive.net/display/Public/QIN-BRAIN-DSC-MRI#21267383b89ada0490e14a66b34d72fe8d9d8a0b) | MRI | segmentation|
| [COVID-19 Detection X-Ray] (https://www.kaggle.com/darshan1504/covid19-detection-xray-dataset) | XRAY | classification |

| [Shenzhen Hospital Chest X-ray Set] 	https://www.kaggle.com/yoctoman/shcxr-lung-mask	| X-Ray
RSNA Bone Age	https://www.kaggle.com/kmader/rsna-bone-age?select=boneage-training-dataset.csv	Hand	X-Ray
Knee Osteoarthritis Dataset with Severity Grading	https://www.kaggle.com/shashwatwork/knee-osteoarthritis-dataset-with-severity	Knee	X-Ray
Finding and Measuring Lungs in CT Data	https://www.kaggle.com/kmader/finding-lungs-in-ct-data?select=lung_stats.csv	Lungs	CT
Brain CT Images with Intracranial Hemorrhage Masks	https://www.kaggle.com/vbookshelf/computed-tomography-ct-images	Brain	CT
Liver and Liver Tumor Segmentation	https://www.kaggle.com/andrewmvd/lits-png?select=lits_test.csv	Liver	CT
Alzheimer's Dataset	https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images	Brain	MRI
Brain MRI Images for Brain Tumor Detection	https://www.kaggle.com/jjprotube/brain-mri-images-for-brain-tumor-detection	Brain	MRI


# **Data preparation instruction**
## Installing requirements
```bash
poetry install 
```
## Creating the dataset
Unfortunally due to the copiright restrictions of the source datasets, we can't share the files directly. To obtain the full dataset you have to download the source datasets yourself and run the preprocessing scripts.

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




Overall the dataset should have ** 882,774** images in **.png** format
* **CT - 500k+
* **X-Ray - 250k+** 
* **MRI - 100k+**



