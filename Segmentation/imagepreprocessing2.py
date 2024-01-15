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

def splitXYPaths(input_path, x_identifier, y_identifier, img_format):
    
    """
    This function recursively iterates through
    `input_path`, extracts and splits the paths of X
    and Y images.
    
    Parameters
    ----------
    input_path : {str}
        The relative (or absolute) path of the folder
        that contains the X and Y images.
    x_identifier : {str}
        A (sub)string that uniquely identifies a path
        that belongs to an X image (as opposed to a Y
        image a.k.a label). i.e. A X image pat is sure
        to contain this substring.
    y_identifier ": {str}
        A (sub)string that uniquely identifies a path
        that belonds to a Y image (label). i.e. A Y
        image path is sure to contain this substring.
    img_format : {str}
        E.g. ".jpg", ".png", etc.
    
    Returns
    -------
    x_paths_list : {list}
        List of X image paths.
    y_paths_list : {list}
        List of Y image paths.
    unidentified_paths_list : {list}
        List of image paths that are neither X or Y
        images.
    """
    
    x_paths_list = []
    y_paths_list = []
    unidentified_paths_list = []
    
    for curdir, dirs, files in os.walk(input_path):
        
        dirs.sort()
        files.sort()
        
        for f in files:
            
            if f.endswith(img_format):
                
                if x_identifier in f:
                    x_paths_list.append(os.path.join(curdir, f))
                elif y_identifier in f:
                    y_paths_list.append(os.path.join(curdir, f))
                else:
                    unidentified_paths_list.append(os.path.join(curdir, f))
    
    return x_paths_list, y_paths_list, unidentified_paths_list


file_paths = "/Users/MayraBerrones/Documents/CBIS-DDSM"
x_paths_list, y_paths_list, _ = splitXYPaths(input_path=file_paths,
                                             x_identifier="FULL",
                                             y_identifier="CROP",
                                             img_format=".dcm")



# Run the above function and store its results in a variable.   
full_file_paths = get_filepaths("/Users/MayraBerrones/Documents/CBIS-DDSM")

selected_paths = []

for f in full_file_paths:
  if f.endswith(".dcm"):
    selected_paths.append(f)

#prueba1 = [Path(x_paths_list[i]).stem for i in range(len(x_paths_list))]

for i in range(len(y_paths_list)):
    print(y_paths_list[i])
    imagedata = pydicom.dcmread(y_paths_list[i])
    img =imagedata.pixel_array
    img = (img - img.min()) / (img.max() - img.min()) * 255.0  
    name = Path(y_paths_list[i]).stem
    cv2.imwrite('/Users/MayraBerrones/Documents/CBIS-DDSM/Crop/{}.png'.format(name), img )

#ds = [cv2.imread(selected_paths[i]) for i in range(len(selected_paths))]
#ds_grayscale = [cv2.cvtColor(ds[i], cv2.COLOR_BGR2GRAY) for i in range(len(selected_paths))]
#patient_id = [Path(selected_paths[i]).stem for i in range(len(selected_paths))]

#print(patient_id)