# UMIE Datasets.

---

This project presents a suite of unified scripts to standardize, preprocess, and integrate 882,774 images from 20 open-source medical imaging datasets, spanning modalities such as X-ray, CT, and MR. The scripts allow for seamless and fast download of a diverse medical data set. We create a unified set of annotations allowing for merging the datasets together without mislabelling. Each dataset is preprocessed with a custom sklearn pipeline. The pipeline steps are reusable across the datasets. The code was designed so that preorocessing a new dataset is simple and requires only reusing the available pipeline steps with customization performed through setting the appropriate values of the pipeline params.

The labels and segmentation masks were unified to be compliant with RadLex ontology.

## Datasets
| uid | Dataset | Modality | TASK |
| :------:| :------:| :--------:|:-------:|
| 0 | [KITS-23](https://kits-challenge.org/kits23/)   | CT | classification/segmentation|
| 1 | [CoronaHack](https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset) | XRAY | classification|
| 2 | [Alzheimers Dataset](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)	| MRI | Classification |
| 3 | [Brain Tumor Classification](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) | MRI | classification|
| 4 | [COVID-19 Detection X-Ray](https://www.kaggle.com/datasets/darshan1504/covid19-detection-xray-dataset) | XRAY | classification |
| 5 | [Finding and Measuring Lungs in CT Data](https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data)	| CT | Segmentation |
| 6 | [Brain CT Images with Intracranial Hemorrhage Masks](https://www.kaggle.com/datasets/vbookshelf/computed-tomography-ct-images)	| CT | Classification |
| 7 | [Liver and Liver Tumor Segmentation](https://www.kaggle.com/datasets/andrewmvd/lits-png)| CT | Classification, Segmentation |
| 8 | [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/jjprotube/brain-mri-images-for-brain-tumor-detection) | MRI | Classification |
| 9 | [Knee Osteoarthritis Dataset with Severity Grading](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity)	|	X-Ray | Classification |
| 10 | [Brain Tumor Progression](https://www.cancerimagingarchive.net/collection/brain-tumor-progression/) | MRI | segmentation|
| 11 | [Chest X-ray 14](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345) | XRAY | classification|
| 12 | [COCA- Coronary Calcium and chest CTs](https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa) | CT | Segmentation |
| 13 | [BrainMetShare](https://aimi.stanford.edu/brainmetshare) | MRI | Segmentation |

