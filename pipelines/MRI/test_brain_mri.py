import PIL
import os
from PIL import Image

if __name__ == "__main__":
    dir_1 = ''  # images source directory
    dir_2 = ''
    dir_d1 = ''  # image target location
    dataset_id = '23'   # dataset identifier

    for filename in os.listdir(dir_1):
        # new name
        n_name = filename.replace('.jpeg', '')
        n_name = n_name.replace('.JPEG', '')
        n_name = n_name.replace('.JPG', '')
        n_name = n_name.replace('.jpg', '')
        n_name = n_name.replace(' ', '')
        n_name = dataset_id + '111' + n_name
        n_name = n_name + "_Brain-Good.png"
        print(filename)
        try:
            image = PIL.Image.open(dir_1 + filename)
            image.save((dir_d1 + n_name), format="png")
        except IOError:
            print('Not found image')

    for filename in os.listdir(dir_2):
        n_name = filename.replace('.jpeg', '')
        n_name = n_name.replace('.JPEG', '')
        n_name = n_name.replace('.JPG', '')
        n_name = n_name.replace('.jpg', '')
        n_name = n_name.replace(' ', '')
        n_name = dataset_id + '222' + n_name
        n_name = n_name + "_Brain-Tumor.png"
        print(filename)
        try:
            image = PIL.Image.open(dir_2 + filename)
            image.save((dir_d1 + n_name), format="png")
        except IOError:
            print('Not found image')
