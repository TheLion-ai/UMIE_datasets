# Datasets


---

## 0.KITS-23

1. Clone the [KITS-23 repository](https://github.com/neheller/kits23).
2. Enter the KITS-23 directory and install the packages with pip.
```bash
cd kits23
pip3 install -e .
```
3. Run the following command to download the data to the `dataset/` folder.
```
kits23_download_data
```
4. Fill in the `source_path` and `target_path` `KITS-23Pipeline()` in `config/runner_config.py`.
 e.g.
```python
    KITS23Pipeline(
        path_args={
            "source_path": "kits23/dataset",  # Path to the dataset directory in KITS23 repo
            "target_path": TARGET_PATH,
            "labels_path": "kits23/dataset/kits23.json",  # Path to kits23.json
        },
        dataset_args=dataset_config.KITS23
    ),
```
---
## 1. Xray CoronaHack -Chest X-Ray-Dataset
1. Go to [CoronaHack](https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset) page on Kaggle.
2. Login to your Kaggle account.
3. Download the data.
4. Extract `archive.zip`.
5. Fill in the `source_path` to the location of the `archive` folder in `CoronaHackPipeline()` in `config/runner_config.py`.

## 2. Alzheimer's Dataset ( 4 class of Images)
1. Go to [Alzheimer's Dataset](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images) page on Kaggle.
2. Login to your Kaggle account.
3. Download the data.
4. Extract `archive.zip`.
5. Fill in the `source_path` to the location of the `archive` folder in `AlzheimersPipeline()` in `config/runner_config.py`.

## 3. Brain Tumor Classification (MRI)
1. Go to [Brain Tumor Classification](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels) page on Kaggle.
2. Login to your Kaggle account.
3. Download the data.
4. Extract `archive.zip`.
5. Fill in the `source_path` to the location of the `archive` folder in `BrainTumorClassificationPipeline()` in `config/runner_config.py`.

## 4. COVID-19 Detection X-Ray
1. Go to [COVID-19 Detection X-Ray](https://www.kaggle.com/datasets/darshan1504/covid19-detection-xray-dataset) page on Kaggle.
2. Login to your Kaggle account.
3. Download the data.
4. Extract `archive.zip`.
5. Fill in the `source_path` to the location of the `archive` folder in `COVID19DetectionPipeline()` in `config/runner_config.py`.

## 5. Finding and Measuring Lungs in CT Data
1.  Go to [Finding and Measuring Lungs in CT Data](https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data) page on Kaggle.
2. Login to your Kaggle account.
3. Download the data.
4. Extract `archive.zip`.
5. Fill in the `source_path` to the location of the `archive/2d_images` folder in `FindingAndMeasuringLungsPipeline()` in `config/runner_config.py`. Fill in `masks_path` with the location of the `archive/2d_masks` folder.

## 6. Brain CT Images with Intracranial Hemorrhage Masks
1. Go to [Brain With Intracranial Hemorrhage](https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images) page on Kaggle.
2. Login to your Kaggle account.
3. Download the data.
4. Extract `archive.zip`.
5. Fill in the `source_path` to the location of the `archive` folder in `BrainWithIntracranialHemorrhagePipeline()` in `config/runner_config.py`. Fill in `masks_path` with the same path as the `source_path`.

## 7. Liver and Liver Tumor Segmentation (LITS)
1. Go to   [Liver and Liver Tumor Segmentation](https://www.kaggle.com/datasets/andrewmvd/lits-png).
2. Login to your Kaggle account.
3. Download the data.
4. Extract `archive.zip`.
5. Fill in the `source_path` to the location of the `archive` folder in `COVID19DetectionPipeline()` in `config/runner_config.py`. Fill in `masks_path` too.

## 8. Brain MRI Images for Brain Tumor Detection
1. Go to [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/jjprotube/brain-mri-images-for-brain-tumor-detection) page on Kaggle.
2. Login to your Kaggle account.
3. Download the data.
4. Extract `archive.zip`.
5. Fill in the `source_path` to the location of the `archive` folder in `BrainTumorDetectionPipeline()` in `config/runner_config.py`.

## 9. Knee Osteoarthrithis Dataset with Severity Grading
1. Go to [Knee Osteoarthritis Dataset with Severity Grading](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity).
2. Login to your Kaggle account.
3. Download the data.
4. Extract `archive.zip`.
5. Fill in the `source_path` to the location of the `archive` folder in `COVID19DetectionPipeline()` in `config/runner_config.py`.

## 10. Brain-Tumor-Progression
1. Go to [Brain Tumor Progression](https://wiki.cancerimagingarchive.net/display/Public/Brain-Tumor-Progression#339481190e2ccc0d07d7455ab87b3ebb625adf48) dataset from the cancer imaging archive.

## 11. Chest X-ray 14
1. Go to [Chest X-ray 14](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345).
2. Create an account.
3. Download the `images` folder and `DataEntry2017_v2020.csv`.

## 12. COCA- Coronary Calcium and chest CTs
1. Go to [COCA- Coronary Calcium and chest CTs](https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa).
2. Log in or sign up for a Stanford AIMI account.
3. Fill in your contact details.
4. Download the data with [azcopy](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10).
5. Fill in the `source_path` with the location of the `cocacoronarycalciumandchestcts-2/Gated_release_final/patient` folder. Fill in `masks_path` with `cocacoronarycalciumandchestcts-2/Gated_release_final/calcium_xml` xml file.

## 13. BrainMetShare
1. Go to [BrainMetShare](https://aimi.stanford.edu/brainmetshare).
2. Log in or sign up for a Stanford AIMI account.
3. Fill in your contact details.
4. Download the data with [azcopy](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10).