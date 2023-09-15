import os
import glob
import cv2
from pathlib import Path

def main_Brain_Tumor_Classification(source_dir, result_dir):

    target_path = os.path.join(result_dir, "Brain_Tumor_Classification")
    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(target_path)
    # path to directory with images
    path = source_dir
    # Extension of images
    extension = '.jpg'
    # Searching for images recursively
    imgs = glob.glob(f"{path}/**/*{extension}", recursive=True)
    print(imgs)

    glioma_tumor = 0
    meningioma_tumor = 0
    no_tumor = 0
    pituitary_tumor = 0

    for img in imgs:
        pat0 = Path(img).resolve().parent
        pat1 = pat0.parent
        # name of tumor
        categoryname = str(pat0)[len(str(pat1))+1:]
        print(categoryname)
        image = cv2.imread(img)

        # creating id's for images with the same label
        if categoryname == 'pituitary_tumor':
            counter = pituitary_tumor
            pituitary_tumor += 1

        if categoryname == 'no_tumor':
            counter = no_tumor
            no_tumor += 1

        if categoryname == 'meningioma_tumor':
            counter = meningioma_tumor
            meningioma_tumor += 1

        if categoryname == 'glioma_tumor':
            counter = glioma_tumor
            glioma_tumor += 1

        if not os.path.exists(os.path.join(target_path, f'{categoryname}')):
            os.makedirs(os.path.join(target_path, f'{categoryname}'))
        cv2.imwrite(os.path.join(target_path, f'{categoryname}') + f'/09{counter}_{categoryname}.png', image)
