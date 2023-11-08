import os


if __name__ == "__main__":
    dir = ''    # path to folder with images
    files = os.listdir(dir)


    for index, file in enumerate(files):
        if (file[:2] != '._'):
            n1 = file
            # modify name of file
            n1 = n1.replace('Healthy', 'Good')
            n1 = n1.replace('_', '')
            n1 = n1[:11] + '_Lungs' + n1[11:]
            # add dataset identifier
            n1 = '06' + n1
            print(n1)
            # rename file
            os.rename(os.path.join(dir, file), os.path.join(dir, n1))


