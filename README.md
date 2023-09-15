# UMIE_datasets
# <center> **UMIE DATASET** </center>
### This branch consists of python modules that:
* Create segmentation masks, with standardized grayscale values
* Change dataset names to standardized form **DatasetID_InternalId_BodyPart-Label.png**

### To use modules you need to specify dataset paths in your local environment
#### Exemplary block of code to change

```
    dir_1 = ''  # images source directory
    dir_2 = ''  # masks source directory
    dir_d1 = '' # destination masks directory
    dir_d2 = '' # destination images directory
```

### To avoid repeating names we added ID prefix to each dataset

We trained our model with medical images from 3 different modalities **CT**, **MRI**, **X-Ray**. Here are datasets that we prepared and used for training:
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


Overall we prepared ** 882,774** images in **.png** format
* **CT - 500k+
* **X-Ray - 250k+** 
* **MRI - 100k+**

### All necessary packages are included in pyproject.toml file.

