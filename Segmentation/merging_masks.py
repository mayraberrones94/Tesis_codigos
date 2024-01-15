import cv2
import numpy as np
import os
import pandas as pd


def get_filepaths(directory):

    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

def merge_masks(mask_paths):
    merged_mask = None
    for mask_path in mask_paths:
        if os.path.isfile(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if merged_mask is None:
                merged_mask = mask
            else:
                merged_mask = cv2.bitwise_or(merged_mask, mask)
    if merged_mask is None:
        return None
    _, merged_mask = cv2.threshold(merged_mask, 128, 255, cv2.THRESH_BINARY)
    return merged_mask



def extract_image_info(image_filename):
    filename = os.path.splitext(image_filename)[0]  # Remove the file extension
    parts = filename.split("_")  # Split the filename by underscores
    
    patient_id = parts[2]  # Extract the patient ID
    img_view = parts[3]  # Extract the image view (e.g., LEFT, RIGHT)
    cc_mlo = parts[4]  # Extract the description (e.g., CC, MLO)
    mask_num = parts[5]  # Extract the mask number as an integer

    return {
        "Patient_ID": patient_id,
        "Image_View": img_view,
        "CC_MLO": cc_mlo,
        "Mask_Number": mask_num
    }

"""

image_folder = "/Users/MayraBerrones/Documents/DDSM-CIBIS/Masks_enhance"
image_files = os.listdir(image_folder)

image_data = []

for image_file in image_files:
    image_info = extract_image_info(image_file)
    image_data.append(image_info)
    #with open("output.txt", "a") as f:
       # print(image_file, file = f)

df = pd.DataFrame(image_data)
df2 = pd.DataFrame(image_files)

datos_full = pd.read_csv('/Users/MayraBerrones/Documents/DDSM-CIBIS/Datos.csv')

print(len(datos_full['id_patiend']))


multiple_mask = []
for index, row in datos_full.iterrows():
    number = row['Mask_num']
    name = row['Full']
    if number != 1:
        multiple_mask.append(name)
        with open("output.txt", "a") as f:
            print(name, file = f)

print(multiple_mask)



"""

name = 'Calc-Training_P_01838_LEFT_MLO_MASK_'
mask_paths = []
for i in range (1,8):
    mask_paths.append(f"/Users/MayraBerrones/Documents/DDSM-CIBIS/Masks_enhance/{name}{i}.png")


# Example usage
#mask_paths = ['Calc-Training_P_00008_LEFT_CC_MASK_1.png', 'Calc-Training_P_00008_LEFT_CC_MASK_2.png', 'Calc-Training_P_00008_LEFT_CC_MASK_3.png']
print(mask_paths)
merged_mask = merge_masks(mask_paths)

cv2.imwrite(f"/Users/MayraBerrones/Documents/DDSM-CIBIS/Masks_full/{name}_Full.png", merged_mask)

