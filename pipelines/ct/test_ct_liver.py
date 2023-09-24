import PIL
import os
from PIL import Image

import yaml
import os
from preprocessing.get_file_paths import GetFilePaths
from preprocessing.add_new_ids import AddNewIds
from preprocessing.convert_nii2dcm import ConvertNii2Dcm
from preprocessing.convert_dcm2png import ConvertDcm2Png
from preprocessing.create_masks_from_xml import CreateMasksFromXML
from preprocessing.create_file_tree import CreateFileTree
from preprocessing.create_blank_masks import CreateBlankMasks
from preprocessing.delete_imgs_without_masks import DeleteImgsWithoutMasks
from sklearn.pipeline import Pipeline


def preprocess_[DATASET_NAME](source_path, target_path):
    dataset_name = ""
    dataset_uid = yaml.load(open('config/dataset_uid_config.yaml'), Loader=yaml.FullLoader)[dataset_name] 
    phases = yaml.load(open('config/phases_config.yaml'), Loader=yaml.FullLoader)[dataset_name]
    mask_encoding_config = yaml.load(open('config/masks_encoding_config.yaml'), Loader=yaml.FullLoader)
    dataset_masks = yaml.load(open('config/dataset_masks_config.yaml'), Loader=yaml.FullLoader)[dataset_name]

    mask_colors_old2new = {v: mask_encoding_config[k] for k, v in dataset_masks.items()}

    params = {
        "target_path": target_path,
        "masks_path": masks_path,
        "dataset_name": dataset_name,
        "dataset_uid": dataset_uid,
        "phases": phases,
        "dataset_masks": dataset_masks,
        "z-fill": 3,
        "img_id_extractor": lambda x: os.path.basename(x).split("-")[-1],
        "study_id_extractor": lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))),
        "mask_colors_old2new": mask_colors_old2new
    }

    pipeline = Pipeline(steps=[
        ("create_file_tree", CreateFileTree(**params)),
        ("convert_nii2dcm", ConvertNii2Dcm(**params)), # TODO: add conversion from nii to dcm
        ("get_file_paths", GetFilePaths(**params)),
        ("add_new_ids", AddNewIds(**params)),
        ("convert_dcm2png", ConvertDcm2Png(**params)),
        # Choose either to create blank masks or delete images without masks
        # ("create_blank_masks", CreateBlankMasks(**params)),
        ("delete_imgs_without_masks", DeleteImgsWithoutMasks(**params))
    ], )
    pipeline.transform(X=source_path)

if __name__ == "__main__":
    dir_1 = ''  # images source directory
    dir_2 = ''   # masks source
    dir_d1 = '' # image target location
    dir_d2 = ''    # liver masks target
    dir_d3 = ''    # liver lesion mask target
    dataset_id = ''   # dataset identifier


    for filename in os.listdir(dir_1):
        filename = filename.replace('._', '')
        image_n = filename.replace('segmentation', 'volume')
        image_n = image_n.replace('_lesionmask', '')
        mask_liver = filename.replace('_lesionmask', '_livermask')

        n_name = filename.replace('_lesionmask', '')
        n_name = n_name.replace('_', '')
        n_name = n_name.replace('-', '')
        n_name = n_name.replace('.png', '')
        n_name = dataset_id + n_name

        print(filename)
        try:
            image = PIL.Image.open(dir_1 + image_n)
            mask_l = PIL.Image.open(dir_2 + mask_liver)
            mask_t = PIL.Image.open(dir_2 + filename)
            if not mask_t.getbbox():
                n_name = n_name + '_Liver-Good.png'
            else:
                n_name = n_name + '_Liver-Lesion.png'

            image.save((dir_d1 + n_name), format="png")
            # changing masks pixels values range from 0-1 to 0-255
            r, g, b = mask_l.split()
            r = r.point(lambda i: i * 255)
            g = g.point(lambda i: i * 255)
            b = b.point(lambda i: i * 255)
            mask_l = Image.merge('RGB', (r, g, b))
            mask_l.save((dir_d2 + n_name), format="png")
            r, g, b = mask_t.split()
            r = r.point(lambda i: i * 255)
            g = g.point(lambda i: i * 255)
            b = b.point(lambda i: i * 255)
            mask_t = Image.merge('RGB', (r, g, b))
            mask_t.save((dir_d3 + n_name), format="png")
        except IOError:
            print('Not found image')

