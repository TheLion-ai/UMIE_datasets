from tcia_utils import nbia_v4 as nbia
import csv 
from pathlib import Path
import re
import os
import shutil
from typing import Optional, Callable

class Dataset:
  def __init__(
    self,
    name: str,                          # Required
    manifest_src: Optional[str] = None,  # Optional
    path: Optional[str] = None,         # Optional
    csv: Optional[str] = None,          # Optional
    callback: Optional[Callable] = None # Optional
  ):
    self.name = name
    self.manifest_src = manifest_src
    self.path = path
    self.csv = csv
    self.callback = callback

  def __repr__(self):
    return (f"Dataset(name={self.name!r}, manifest_src={self.manifest_src!r}, "
            f"path={self.path!r}, csv={self.csv!r}, callback={self.callback!r})")
  

  
  
  
datasets = [
  Dataset(
    name="LIDC_MANIFEST",
    manifest_src="./nbia_manifests/TCIA_LIDC-IDRI_20200921.tcia",
    path="./datasets/LIDC",
    csv="LIDC_metadata"

    # Data Citation
    # Armato III, S. G., McLennan, G., Bidaut, L., McNitt-Gray, M. F., Meyer, C. R., Reeves, A. P., Zhao, B., Aberle, D. R., Henschke, C. I., Hoffman, E. A., Kazerooni, E. A., MacMahon, H., Van Beek, E. J. R., Yankelevitz, D., Biancardi, A. M., Bland, P. H., Brown, M. S., Engelmann, R. M., Laderach, G. E., Max, D., Pais, R. C. , Qing, D. P. Y. , Roberts, R. Y., Smith, A. R., Starkey, A., Batra, P., Caligiuri, P., Farooqi, A., Gladish, G. W., Jude, C. M., Munden, R. F., Petkovska, I., Quint, L. E., Schwartz, L. H., Sundaram, B., Dodd, L. E., Fenimore, C., Gur, D., Petrick, N., Freymann, J., Kirby, J., Hughes, B., Casteele, A. V., Gupte, S., Sallam, M., Heath, M. D., Kuhn, M. H., Dharaiya, E., Burns, R., Fryd, D. S., Salganicoff, M., Anand, V., Shreter, U., Vastagh, S., Croft, B. Y., Clarke, L. P. (2015). Data From LIDC-IDRI [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX

  ),
  Dataset(
    name="LIDC_XML"
  ),
  Dataset(
    name="CMMD_MANIFEST",
    manifest_src="./nbia_manifests/The-Chinese-Mammography-Database.tcia",
    path="./datasets/CMMD",
    csv="CMMD_metadata"
    
    # Data Citation
    #         ui, Chunyan; Li Li; Cai, Hongmin; Fan, Zhihao; Zhang, Ling; Dan, Tingting; Li, Jiao; Wang, Jinghua. (2021) The Chinese Mammography Database (CMMD): An online mammography database with biopsy confirmed types for machine diagnosis of breast. The Cancer Imaging Archive. DOI: https://doi.org/10.7937/tcia.eqde-4b16
  ),
  Dataset(
    name="CMMD_CLINICAL"
  ),
  
]

def add_padding(input_string):
  while input_string.endswith('0'):
    input_string = input_string[:-1]
    
  return input_string + "00000"

def copy_all_files(src_folder, dst_folder):
  for filename in os.listdir(src_folder):

    src_file = os.path.join(src_folder, filename)
    dst_file = os.path.join(dst_folder, filename)

    # # Copy only if it's a file (skip subfolders)
    if os.path.isfile(src_file):
      shutil.copy2(src_file, dst_file) 
          

config_file = "./.pipeline.env"
with open(config_file, "a") as file:
  
  for dt in datasets:
    manifest_src = ""
  
   
    if dt.manifest_src is not None and dt.path is not None:

      output = Path(dt.path)
      output.mkdir(parents=True, exist_ok=True)
      

      manifest_output = Path(dt.path + "/manifest_download")
      manifest_output.mkdir(exist_ok=True)
      dataframe = nbia.downloadSeries(dt.manifest_src, 
                               input_type="manifest", 
                               format="csv", 
                               csv_filename=dt.path + "/" + dt.csv, 
                               path=dt.path+"/manifest_download")
      manifest_src = dt.path + "/out"
      outdir = Path(manifest_src)
      outdir.mkdir(exist_ok=True)
      with open(dt.path + "/" + dt.csv + ".csv", mode="r", newline="") as rfile:
        reader = csv.DictReader(rfile)
        for row in reader:
          src_folder = Path(dt.path + "/manifest_download/" + row["SeriesInstanceUID"])
          if src_folder.is_dir():

            patientId = row["PatientID"]
            
            studyDate = row["StudyDate"]
            studyId = "NA" if row["Study ID"] == "" else str(int(float(row["Study ID"])))
            studyDesc = "NA" if row["Study Description"] == "" else row["Study Description"][:54]
            studyDesc = re.sub(r"[^a-zA-Z0-9]+", ' ', studyDesc)
            studyInstanceUID = row["StudyInstanceUID"][-5:]
            
            seriesNumb = add_padding(row["Series Number"])
            seriesDesc = "NA" if row["SeriesDescription"] == "" else row["SeriesDescription"][:54]
            seriesDesc = re.sub(r"[^a-zA-Z0-9]+", ' ', seriesDesc)
            seriesInstanceUID = row["SeriesInstanceUID"][-5:]
            
            new_path = f"{manifest_src}/{patientId}/{studyDate}-{studyId}-{studyDesc}-{studyInstanceUID}/{seriesNumb}-{seriesDesc}-{seriesInstanceUID}"
            path_dst_folder = Path(new_path)
            path_dst_folder.mkdir(parents=True, exist_ok=True)
            
            old_path = f"{dt.path}/manifest_download/{row["SeriesInstanceUID"]}"
 
            copy_all_files(old_path, new_path)
            
          else:
            print('folder Do not exists->' + row["SeriesInstanceUID"])
            
        rfile.close()
          
    file.write(f"{dt.name}={os.getcwd()}{manifest_src.replace(".", "")}\n")
  
  file.close()