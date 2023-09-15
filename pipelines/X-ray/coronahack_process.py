import PIL
import os

def main_coronahack(source_dir, result_dir):

    target_path = os.path.join(result_dir, "Coronahack")
    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(target_path)
    dir = []    # all directories of source images
    dir.append(source_dir + '/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/')
    dir.append(source_dir + '/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/')
    dir_csv = source_dir + '/Chest_xray_Corona_Metadata.csv'    # path to .csv file with labels
    dir_d = target_path      # path to destination folder
    labels = ['Virus', 'bacteria']      # unified labels contained in dataset
    dataset_id = '05'   # identifier of dataset

    with open(dir_csv) as infile:
        lines = infile.readlines()[1:]
        # for each line in .csv file open, process and save image
        for line in lines:
            data1 = line.split(',')
            data1[len(data1)-1] = data1[len(data1)-1][:-1]
            # extract file name
            f_name = data1[1]
            # extract identifier of image without - and _
            namep = f_name.replace('_', '')
            namep = namep.replace('-', '')
            namep = namep.replace('.jpg', '')
            namep = namep.replace('.jpeg', '')
            n_name = dataset_id + namep + '_' + 'Lungs' + '-'       # firfile namest part of new
            print(n_name)
            for i in range(len(dir)):       # look for image in every source destination
                dir_1 = dir[i]
                # open image
                try:
                    image = PIL.Image.open(dir_1 + f_name)
                    label1 = data1[2]
                    # add appropriate label and save image
                    if(label1 == 'Normal'):
                        n_name += 'Good.png'
                        image.save((dir_d + '/' + n_name), format="png")
                    else:
                        label2 = data1[5]
                        if(label2 == 'Virus'):
                            n_name += 'PneumoniaVirus.png'
                            image.save((dir_d + '/' + n_name), format="png")
                        elif(label2 == 'bacteria'):
                            n_name += 'PneumoniaBacteria.png'
                            image.save((dir_d + '/' + n_name), format="png")
                except IOError:
                    print('N')


