# **Universal medical image encoder - Data management plan (DMP)**

## 1. Data collection and documentation

### **1.1. Purpose of data collection**
Collected data will be used to train neural network for task of classification and
segmentation medical images. Main usage of it will be detection of abnormalities and
diseases, which would improve quality of patient’s diagnosis.
### **1.2. What data will you collect, observe, generate or reuse?**
This project will reuse:
* CT images
* MRI images
* X-ray images
* information about type of disease, malignancy, etc.
* description of abnormalities location
  
Only images with information about disorder type (eg. healthy, cancer, pneumonia, etc.) or
with information about location of abnormality on image will be useful for our project.
<br/><br/>

This part of project will generate:
* processed imaging data in PNG format with standardized, unique names.
* grayscale masks in PNG format for part of images
* python scripts for images conversion, extracting information and creating masks
<br/><br/>

Formats of collected data:
* DICOM
* PNG
* JPEG
* NIFTY
* XML
* CSV

### **1.3. How will the data be collected, observed or generated?**

The images will be collected from different public repositories, mostly from kaggle and
cancer imaging archive. Full list of resources will be available in separate document.
Imaging and clinical data of patients will be acquired following predefined protocol
standards within the clinical workup. However these standards vary significantly, so
processing is necessary to meet our standard. 
<br/><br/>

## **2. Ethics, legal and security issues**
### **2.1. How will ethical issues be addressed and handled?**
Data available in used resources is completely anonymous. If there are multiple images of
one patient (common in MRI), there is provided a patient ID, so they can be matched, but
there is not publicly available any information that can assign patients personal data to
IDs. Some resources provide information about patient’s age, sex or nationality, but we
don’t use it in our project.
### **2.2. How will you handle copyright issues?**
Available data is open source or covered by creative commons license. We will only use
data, whose owner agrees to use it for non commercial scientific research. List of resources citations will be provided in separate file with project results.
<br/><br/>

## **3. Data storage**
### **3.1. How will your data be stored during the research?**
Data will be downloaded to local computers, processed to meet our standard and then
uploaded to private folder on Onedrive. During the process of neural network training,
prepared data will be downloaded in real time.

