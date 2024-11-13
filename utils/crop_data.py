"""Crop the kidney tumor images to a square region containing the kidney with tumor."""

import glob
import os

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, center_of_mass, label
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects


def crop_kidney_tumor(mask: np.array) -> tuple[int, int, int, int] | None:
    """
    Crops the image to a square region containing the kidney with tumor.

    Parameters:
    mask: numpy array with kidney (1) and tumor (2) labels

    Returns:
    tuple: (top, bottom, left, right) coordinates for cropping
    """
    # Separate kidney and tumor masks
    kidney_mask = mask == 1
    tumor_mask = mask == 2

    if not np.any(tumor_mask):
        return None
        # raise ValueError("No tumor found in the mask")

    # Clean up kidney mask
    # 1. Remove small objects (noise)
    min_kidney_size = 100  # adjust this threshold based on your image size
    kidney_mask = remove_small_objects(kidney_mask, min_size=min_kidney_size)

    # 2. Dilate to connect nearby components
    kernel = np.ones((5, 5), bool)  # adjust kernel size if needed
    kidney_mask = binary_dilation(kidney_mask, structure=kernel)

    # 3. Get connected components
    labeled_kidneys, num_components = label(kidney_mask)
    props = regionprops(labeled_kidneys)

    if not props:
        return None
        # raise ValueError("No kidney regions found after preprocessing")

    # Sort regions by area in descending order
    props.sort(key=lambda x: x.area, reverse=True)

    # Get tumor center of mass
    tumor_com = center_of_mass(tumor_mask)

    # Calculate distances from each kidney centroid to tumor
    distances = []
    kidney_bounds = []

    for prop in props:
        centroid = prop.centroid
        distance = np.sqrt((tumor_com[0] - centroid[0]) ** 2 + (tumor_com[1] - centroid[1]) ** 2)
        distances.append(distance)

        # Get bounding box (min_row, min_col, max_row, max_col)
        bbox = prop.bbox
        kidney_bounds.append((bbox[0], bbox[2], bbox[1], bbox[3]))  # top, bottom  # left, right

    # Select kidney closest to tumor
    target_kidney_idx = np.argmin(distances)
    top, bottom, left, right = kidney_bounds[target_kidney_idx]

    # Include tumor in bounds
    tumor_rows, tumor_cols = np.where(tumor_mask)
    if len(tumor_rows) > 0 and len(tumor_cols) > 0:
        top = min(top, np.min(tumor_rows))
        bottom = max(bottom, np.max(tumor_rows))
        left = min(left, np.min(tumor_cols))
        right = max(right, np.max(tumor_cols))

    # Add margin
    margin = 10
    height = bottom - top + 2 * margin
    width = right - left + 2 * margin

    # Make the crop square by taking the larger dimension
    size = max(width, height)

    # Calculate new bounds ensuring they're centered
    center_y = (top + bottom) / 2
    center_x = (left + right) / 2

    top = int(center_y - size / 2)
    bottom = int(center_y + size / 2)
    left = int(center_x - size / 2)
    right = int(center_x + size / 2)

    # Ensure bounds are within image
    top = max(0, top)
    bottom = min(mask.shape[0], bottom)
    left = max(0, left)
    right = min(mask.shape[1], right)

    # Final adjustment to ensure square output
    final_size = min(bottom - top, right - left)
    center_y = (top + bottom) / 2
    center_x = (left + right) / 2

    top = int(center_y - final_size / 2)
    bottom = int(center_y + final_size / 2)
    left = int(center_x - final_size / 2)
    right = int(center_x + final_size / 2)

    # Final boundary check
    top = max(0, top)
    bottom = min(mask.shape[0], bottom)
    left = max(0, left)
    right = min(mask.shape[1], right)

    return top, bottom, left, right


def apply_crop(image: np.array, crop_coords: tuple[int, int, int, int]) -> np.array:
    """
    Apply the calculated crop coordinates to an image.

    Parameters:
    image: numpy array of the image to crop
    crop_coords: tuple of (top, bottom, left, right)

    Returns:
    numpy array: cropped image
    """
    top, bottom, left, right = 0, 0, 0, 0
    top, bottom, left, right = crop_coords
    return image[top:bottom, left:right]


img_paths = glob.glob(os.path.join("**/cropped_data/00_kits23/CT/Images/*.png"), recursive=True)
for img_path in img_paths:
    mask_path = img_path.replace("Images", "Masks")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(img_path)
    crop_coords = crop_kidney_tumor(mask)
    if crop_coords is None:
        continue
    cropped_image = apply_crop(original_image, crop_coords)
    cropped_mask = apply_crop(mask, crop_coords)
    np.place(mask, mask == 1, 125)
    np.place(mask, mask == 2, 255)
    new_path = f"cropped_data/cropped_kits23/{os.path.basename(img_path)}"
    # new_mask_path = f"cropped_data/{os.path.basename(mask_path)[:-4]}_mask.png"
    # new_img_path = f"cropped_data/{os.path.basename(img_path)[:-4]}_org.png"
    cv2.imwrite(new_path, cropped_image)
    # cv2.imwrite(new_mask_path, mask)
    # cv2.imwrite(new_img_path, original_image)
