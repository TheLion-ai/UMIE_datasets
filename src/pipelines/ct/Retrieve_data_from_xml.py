import csv
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import PIL.Image
from PIL import Image, ImageDraw


def polygon_to_mask(img_size, polygon):
    # Takes polygon points as list of tuples (x, y) and creates np.array of size(x, y)
    # if pixel is inside polygon value of pixel is set to 1

    img = Image.new("L", img_size, 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    return np.array(img)


class roi_class:
    def __init__(self):
        self.points = []
        self.inclusion = False
        self.mal = 0  # negative mal means no nodule


class maskc:
    def __init__(self):
        self.mask_area = []
        self.incl = True
        self.malignacy = 0


def make_mask(tree, dir1):
    # this dictionary keeps all of our regions of interest for a particular XML file
    rois = {}
    root = tree.getroot()
    # this script allows us to retrieve only the data that we want from the XML file
    for child_of_root in root:
        if child_of_root.tag == "{http://www.nih.gov}readingSession":
            for child_of_session in child_of_root:
                if child_of_session.tag == "{http://www.nih.gov}unblindedReadNodule":
                    mal = 0
                    for roi in child_of_session:
                        if roi.tag == "{http://www.nih.gov}characteristics":
                            for char in roi:
                                if char.tag == "{http://www.nih.gov}malignancy":
                                    mal = char.text
                        if roi.tag == "{http://www.nih.gov}roi":
                            roi_object = roi_class()
                            name = ""
                            roi_object.mal = int(mal)
                            for child in roi:
                                # here we iterate over every child of region of interest and get the values we care about
                                if child.tag == "{http://www.nih.gov}imageSOP_UID":
                                    name = child.text
                                if child.tag == "{http://www.nih.gov}inclusion":
                                    if child.text == "TRUE":
                                        roi_object.inclusion = True
                                if child.tag == "{http://www.nih.gov}edgeMap":
                                    x_cord = int(child[0].text)
                                    y_cord = int(child[1].text)
                                    wspolrzedna = (x_cord, y_cord)
                                    roi_object.points.append(wspolrzedna)
                            rois[roi_object] = name
            # extract names of non nodule images (give it malignacy = -1)
            for child_of_session in child_of_root:
                if child_of_session.tag == "{http://www.nih.gov}nonNodule":
                    roi_object = roi_class()
                    name = ""
                    roi_object.mal = -1
                    for child in child_of_session:
                        if child.tag == "{http://www.nih.gov}imageSOP_UID":
                            name = child.text
                        if child.tag == "{http://www.nih.gov}inclusion":
                            if child.text == "TRUE":
                                roi_object.inclusion = True
                        if child.tag == "{http://www.nih.gov}edgeMap":
                            x_cord = int(child[0].text)
                            y_cord = int(child[1].text)
                            wspolrzedna = (x_cord, y_cord)
                            roi_object.points.append(wspolrzedna)
                    rois[roi_object] = name

    masks = {}
    # lists of names of images with big nodules, small nodules and no nodules
    b_nodules = []
    s_nodules = []
    n_nodules = []

    # divide images into arrays by size of nodule and make dictionary of masks
    for mask_p, name in rois.items():
        if len(mask_p.points) > 1:
            mask1 = maskc()
            width = 512
            height = 512
            mask1.mask_area = polygon_to_mask((width, height), mask_p.points)
            mask1.incl = mask_p.inclusion
            mask1.malignacy = mask_p.mal
            masks[mask1] = name
            b_nodules.append(name)  # big nodules
        elif mask_p.mal == 0:  # small nodules
            s_nodules.append(name)
        elif mask_p.mal == -1:  # no nodules
            n_nodules.append(name)

    # create dictionary of masks where the key is image
    masks_for_images = {}
    for key, value in masks.items():
        masks_for_images.setdefault(value, []).append(key)

    num_mask = 0

    # for every image is calculated average mask and saved to .csv
    # last line in every .csv file is image ID
    # names of .csv files can be changed to be the same as .png images
    for key, image_masks in masks_for_images.items():
        num_mask += 1
        mask_v = np.zeros(
            (len(image_masks[0].mask_area), len(image_masks[0].mask_area[0]))
        )
        mal = 0
        # calculate sum of all masks for the image (evaluations from different radiologists)
        for mask in image_masks:
            mal += mask.malignacy
            if mask.incl == True:
                mask_v += mask.mask_area
            else:
                mask_v -= mask.mask_area
        # calculate average mask
        mal = mal / len(image_masks)
        mask_v = (mask_v > len(image_masks) / 2).astype(int)

        name = key.replace(".", "-")
        name += "_" + str(mal)
        # 0-1 to 0-255 range
        I8 = (((mask_v - mask_v.min()) / (mask_v.max() - mask_v.min())) * 255.9).astype(
            np.uint8
        )
        # convert array to image and save mask
        img = Image.fromarray(I8)
        path1 = os.path.join(dir1, "masks")
        if not os.path.exists(os.path.join(path1)):
            os.makedirs(path1)
        img.save(path1 + "/" + name + ".png")

    # make list of nodules grouped into 3 files by size
    b_nodules = list(dict.fromkeys(b_nodules))
    s_nodules = list(dict.fromkeys(s_nodules))
    n_nodules = list(dict.fromkeys(n_nodules))

    with open((dir1 + "nodules" + ".csv"), "a", newline="") as write_obj:
        csv_writer = csv.writer(write_obj)
        for name in b_nodules:
            csv_writer.writerow(name)

    with open((dir1 + "small_nodules" + ".csv"), "a", newline="") as write_obj:
        csv_writer = csv.writer(write_obj)
        for name in s_nodules:
            csv_writer.writerow(name)

    with open((dir1 + "non_nodules" + ".csv"), "a", newline="") as write_obj:
        csv_writer = csv.writer(write_obj)
        for name in n_nodules:
            csv_writer.writerow(name)


def retrieve_data_main(source_dir, result_dir):
    target_path = os.path.join(result_dir, "LIDC")
    if not os.path.exists(os.path.join(target_path)):
        os.makedirs(target_path)
    dir = source_dir  # source directory
    pliki = glob.glob(dir + "*.xml", recursive=True)

    # directory where results (masks and lists) will be saved
    dir_result = target_path

    # processing all files in directory
    for file in pliki:
        print("Processing file: ", file)
        tree = ET.parse(file)
        make_mask(tree, dir_result)

    print("Processing finished")
