import PIL
import os
from PIL import Image

if __name__ == "__main__":
    dir_1 = ''  # images source directory
    dir_2 = '' # masks source
    dir_d1 = ''
    dir_d2 = ''
    dataset_id = '16'   # dataset identifier

    for filename in os.listdir(dir_1):
            print(filename)
            # new file name
            n_name = filename
            n_name = n_name.replace('.png', '')
            n_name = n_name.replace('.jpeg', '')
            n_name = dataset_id + n_name
            n_name = n_name + '_Lungs.png'
            print(n_name)
            mask_n = filename
            mask_n = mask_n.replace('.png', '')
            mask_n = mask_n + '_mask.png'
            try:
                image = PIL.Image.open(dir_1 + filename)
                mask = PIL.Image.open(dir_2 + mask_n)
                image.save((dir_d1 + n_name), format="png")
                mask.save((dir_d2 + n_name), format="png")
            except IOError:
                print('Not found image')

