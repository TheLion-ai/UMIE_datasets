import PIL
import os
from PIL import Image

if __name__ == "__main__":
    dir_1 = ''  # images source directory
    dir_2 = ''   # masks source
    dir_d1 = ''  # image target location
    dir_d2 = ''  # masks target
    dataset_id = ''   # dataset identifier

    for filename in os.listdir(dir_1):
        filename = filename.replace('._', '')
        # changing .tif to .tiff, so images will be readable for PIL
        n_name = filename.replace('.tif', '.tiff')
        os.rename(dir_1 + filename, dir_1 + n_name)
        os.rename(dir_2 + filename, dir_2 + n_name)
        print(n_name)

    for filename in os.listdir(dir_1):
            filename = filename.replace('._', '')
            # new file name
            n_name = filename
            n_name = n_name.replace('.png', '')
            n_name = n_name.replace('.tiff', '')
            n_name = dataset_id + n_name
            n_name = n_name + '_Lungs.png'
            print(n_name)
            mask_n = filename
            print(filename)
            try:
                image = PIL.Image.open(dir_1 + filename)
                mask = PIL.Image.open(dir_2 + mask_n)
                image.thumbnail(image.size)
                image = image.convert('RGB')
                image.save((dir_d1 + n_name), format="png")
                mask.save((dir_d2 + n_name), format="png")
            except IOError:
                print('Not found image')
