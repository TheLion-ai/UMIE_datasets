import PIL
import os
from PIL import Image

if __name__ == "__main__":
    dir_1 = ''  # images source directory
    dir_d1 = '' # destination images directory
    dataset_id = '15'   # dataset identifier

    l = 0
    for filename in os.listdir(dir_1):
        if(l < 200):
            print(filename)
            # new file name
            n_name = filename
            n_name = n_name.replace('.png', '')
            n_name = n_name.replace('.jpeg', '')
            n_name = '15' + n_name
            n_name = n_name + '_Lungs-ViralPneumonia.png'
            print(n_name)
            try:
                image = PIL.Image.open(dir_1 + filename)
                image.save((dir_d1 + n_name), format="png")
                l += 1

            except IOError:
                print('Not found image')


