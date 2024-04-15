By Murtadha Hssayeni, mhssayeni2017@fau.edu 8/13/2018

Dataset name: Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation
Authors: Murtadha D. Hssayeni, M.S., Muayad S. Croock, Ph.D., Aymen Al-Ani, Ph.D., Hassan Falah Al-khafaji, M.D. and Zakaria A. Yahya, M.D.


A retrospective study was designed to collect head CT scans of subjects with TBI and it was approved by the research and ethics board in the Iraqi ministry of health-Babil Office. CT scans were collected between February and August 2018 from Al Hilla Teaching Hospital-Iraq. Patients information were anonymized. A dataset of 82 CT scans was collected that included 36 scans for patients diagnosed with intracranial hemorrhage with the following types (Intraventricular, Intraparenchymal, Subarachnoid, Epidural and Subdural). Each CT scan for each patient includes about 30 slices with 5 mm slice-thickness. The mean and std of patients' age are 27.8 and 19.5, respectively. 46 of the patients are males and 36 of them are females. Each slice of non-contrast CT scans was annotated by two radiologists by recording hemorrhage types if hemorrhage occurred, and fracture and also the radiologists delineated the ICH regions in each slice. There was a consensus between the radiologists. They did not have access to clinical history of the patients, thus they have the same given data (CT scans) for each subject as the developed algorithm to make their diagnosis.

During data collection, syngo by Siemens Medical Solutions was first used to read the CT DICOM files and save two videos (avi format) using brain and bone windows, respectively. Second, a custom tool was implemented in Matlab and used to read the avi files, record the radiologist annotations, delineate hemorrhage region and save it as white region in a black 650x650 image (jpg format) and then save grey-scale 650x650 images (jpg format) for each CT slice with the brain and bone windows.



Files and folders:
patient_demographics.csv - patient age and gender
hemmorhage_diagnosis.csv - contains the labels (hemorrhage type, fracture) for each slice and for each patient.
Patients_CT - folder


Patients_CT folder includes:
	-The Sub-folder names represent the patients numbers in "patient_demographics.csv"
	-Each sub-folder contains:
		-brain folder
			-all the brain-window CT slices for each patient, numbers of CT slices started from 1.
			-the segmentation of the slices that have hemorrhage (named #HGE_Seg).
		-bone folder
			-all the bone-window CT slices for each patient, numbers of CT slices started from 1.

split_data.py is a python code to load the grayscale images of brain-window of each subject in Patient_CT folder, resize and save them to one folder (tarin\image) and save their segmentation to another folder (train\label).

ct_ich.yml is a conda environment file, allowing creation of a virtual environment for the above Python code. It can be run using conda as follows:

conda env create -f ct_ich.yml
