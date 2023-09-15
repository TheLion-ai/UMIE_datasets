import csv
import ast
import os


# This module is written for ChestX-ray14 dataset
def convert_labels(original_label, condition_names):
    # in original csv file there are more than single condition to one photo
    # and this function converts labels to binary vector [0 , 1, 0, 0 ...]
    original_label = original_label.lower()
    conditions = original_label.split('|')

    label = [0] * len(condition_names)
    for condition in conditions:
        try:
            label[condition_names.index(condition)] = 1
        except ValueError:
            print("Nie ma takiej zmainy na liscie")
    return label


def convert_csv_xray_14(source_dir, result_dir):

    target_path = os.path.join(result_dir, "Chest_X-ray14")
    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(target_path)
    condition_names = ["No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",  "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
    condition_names = [x.lower() for x in condition_names]
    data_csv = source_dir + '/other/Data_Entry_2017_v2020.csv'
    with open((target_path+'/labels.csv'), newline='', mode="w") as write_file:
        writer = csv.writer(write_file, delimiter=',')
        with open(data_csv, newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                img_id = row[0]
                img_label = convert_labels(row[1], condition_names)
                writer.writerow([img_id, img_label])

