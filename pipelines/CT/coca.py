from preprocess import AddNewIds, ConvertDcm2Png, CreateMasksFromXML,  CreateFileTree

def preprocess_coca(source_path: str, target_path: str):

  pipeline = PreprocessPipeline(steps=[
    ("add_new_ids", AddNewIds())
    ("convert_dcm2png", ConvertDcm2Png())
    ("create_masks_from_png", CreateMasksFromXML())
    ("create_file_tree", CreateFileTree())
  ])

  pipeline.fit_transform(source_path, target_path)

if __name__=="__main__":
  source_path = ""
  target_path = ""
  preprocess_coca(source_path, target_path)
  
