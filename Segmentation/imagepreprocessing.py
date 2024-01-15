import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import cv2
import pydicom
import os
import skimage

from PIL import Image

from pathlib import Path

def ShowHist255(img, ignore_zero = False):
    hist, bin_edges = np.histogram(img, bins=255, density=False)
    
    if ignore_zero:
        plt.plot(bin_edges[1:-1], hist[1:])
    else:
        plt.plot(bin_edges[0:-1], hist)
        
    plt.show()

import os

def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

# Run the above function and store its results in a variable.   
full_file_paths = get_filepaths("/Users/MayraBerrones/Documents/test")

selected_paths = []

for f in full_file_paths:
  if f.endswith(".dcm"):
    selected_paths.append(f)
    imagedata = pydicom.dcmread(f)
    img =imagedata.pixel_array
    img = (img - img.min()) / (img.max() - img.min()) * 255.0  
    name = Path(f).stem
    cv2.imwrite('/Users/MayraBerrones/Documents/test/output/{}.png'.format(name), img )

#ds = [cv2.imread(selected_paths[i]) for i in range(len(selected_paths))]
#ds_grayscale = [cv2.cvtColor(ds[i], cv2.COLOR_BGR2GRAY) for i in range(len(selected_paths))]
patient_id = [Path(selected_paths[i]).stem for i in range(len(selected_paths))]

print(patient_id)