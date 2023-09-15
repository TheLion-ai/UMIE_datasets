import PIL
import os
from PIL import Image

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

