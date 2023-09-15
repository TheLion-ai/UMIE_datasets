import PIL
import os
from PIL import Image
import csv

if __name__ == "__main__":
    dir_csv = ''  #csv file path
    dir_1 = ''  # images source directory
    dir_2 = ''   # masks source
    dir_d1 = ''     # images target directory
    dir_d2 = ''      # masks target directory
    dataset_id = ''   # dataset identifier

    with open(dir_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                folder = str(row[0])
                folder = folder.replace('\'', '')
                image = str(row[1])
                image = image.replace('\'', '')
                path_1 = dir_1 + folder + '/bone/' + image + '.jpg'
                path_2 = dir_2 + folder + '/brain/' + image + '.jpg'
                n_name = dataset_id
                if(row[7] == 1):
                    n_name = n_name + folder + image + '_Brain-Good.png'
                else:
                    n_name = n_name + folder + image + '_Brain-Hemorrhage.png'
                print(n_name)
                print(path_1)
                print(path_2)
                try:
                    image = PIL.Image.open(path_1)
                    mask = PIL.Image.open(path_2)
                    image.save((dir_d1 + n_name), format="png")
                    mask.save((dir_d2 + n_name), format="png")
                except IOError:
                    print('Not found image')

                line_count += 1
        print(f'Processed {line_count} lines.')


