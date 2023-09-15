from preprocess import AddNewIds, ConvertDcm2Png, CreateMasksFromPng,  CreateFileTree

def preprocess_brain_met_share(source_path: str, target_path: str):

  pipeline = PreprocessPipeline(steps=[
    ("add_new_ids", AddNewIds())
    ("convert_dcm2png", ConvertDcm2Png())
    ("create_masks_from_png", CreateMasksFromPng())
    ("create_file_tree", CreateFileTree())
  ])

  pipeline.fit_transform(source_path, target_path)

if __name__=="__main__":
  source_path = ""
  target_path = ""
  preprocess_brain_met_share(source_path, target_path)
  
