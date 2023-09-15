# changing names of files to standarise them
import PIL
import os

if __name__ == "__main__":
    dir_1 = ''  # images source directory
    dir_2 = ''  # masks source directory
    dir_d1 = '' # destination masks directory
    dir_d2 = '' # destination images directory
    dataset_id = '10'   # dataset identifier

    for filename in os.listdir(dir_1):
        filename = filename.replace('._', '')
        print(filename)
        # get mask name from file name
        m_name = filename.replace('T1post', '')
        m_name = m_name.replace('.png', '')
        m_name = m_name + 'Mask_Tumor.png'
        print(m_name)
        # first part of new name
        n_name = filename.replace('.png', '')
        n_name = n_name.replace('T1post', '')
        n_name = n_name.replace('_', '')
        n_name = n_name.replace('-', '')
        n_name = dataset_id + n_name
        # open file
        try:
            image = PIL.Image.open(dir_1 + filename)
            try:
                mask = PIL.Image.open(dir_2 + m_name)
                # check if mask is empty (non empty mask means tumor)
                if not mask.getbbox():
                    n_name = n_name + '_Brain.png'
                else:
                    n_name = n_name + '_Brain-Tumor.png'
                mask.save((dir_d1 + n_name), format="png")
                image.save((dir_d2 + n_name), format="png")
            except IOError:
                print('Not found mask')

        except IOError:
            print('Not found image')
