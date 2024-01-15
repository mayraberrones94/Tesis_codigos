import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import cv2
import pydicom
import os
import skimage

from numpy.lib.arraypad import pad

from PIL import Image

from pathlib import Path

def ShowHist255(img, ignore_zero = False):
    hist, bin_edges = np.histogram(img, bins=255, density=False)
    
    if ignore_zero:
        plt.plot(bin_edges[1:-1], hist[1:])
    else:
        plt.plot(bin_edges[0:-1], hist)
        
    plt.show()

def get_filepaths(directory):

    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

def CropBorders(img):
    nrows, ncols = img.shape

    # Get the start and end rows and columns
    l_crop = int(ncols * 0.01)
    r_crop = int(ncols * (1 - 0.04))
    u_crop = int(nrows * 0.01)
    d_crop = int(nrows * (1 - 0.04))

    cropped_img = img[u_crop:d_crop, l_crop:r_crop]
    
    return cropped_img

def Binarisation(img, maxval, show=False):
    
    # First convert image to uint8.
    img = skimage.img_as_ubyte(img)
    
    thresh, th1 = cv2.threshold(src=img,thresh=0.1,maxval=maxval,type=cv2.THRESH_BINARY) # Global thresholding

    otsu_thresh, th2 = cv2.threshold(src=img, thresh=0, maxval=maxval, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Otsu's thresholding

    th3 = cv2.adaptiveThreshold(src=img, maxValue=maxval,adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=9, C=-1)
    
    th4 = cv2.adaptiveThreshold(src=img, maxValue=maxval, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=9, C=-1)

    images = [img, th1, th2, th3, th4]
    titles = ['Original Image',
              'Global Thresholding (v = 0.1)',
              'Global Thresholding (otsu)',
              'Adaptive Mean Thresholding',
              'Adaptive Gaussian Thresholding']

    
    # --- Plot the different thresholds ---
    if show:
        fig, ax = plt.subplots(nrows=1, ncols=5, figsize = (22, 5))

        for i in range(5):
            ax[i].imshow(images[i], cmap="gray")
            ax[i].set_title(titles[i])
        plt.show()
    
    return th1, th2, th3, th4

def DilateMask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    
    return dilated_mask

def ApplyMask(img, dilated_mask):
    result = img.copy()
    result[dilated_mask == 0] = 0 # We only keep areas that are white (255)
    
    return result

def OwnGlobalBinarise(img, thresh, maxval):

    binarised_img = np.zeros(img.shape, np.uint8)
    binarised_img[img >= thresh] = maxval
    
    return binarised_img

def OpenMask(mask, ksize=(23, 23), operation="open"):
        
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=ksize)
    
    if operation == "open":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Then dilate
    edited_mask = cv2.morphologyEx(edited_mask, cv2.MORPH_DILATE, kernel)
    
    return edited_mask

def SortContoursByArea(contours, reverse=True):
    # Sort contours based on contour area.
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Construct the list of corresponding bounding boxes.
    bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours]
    
    return sorted_contours, bounding_boxes

def DrawContourID(img, bounding_box, contour_id):
    # Center of bounding_rect.
    x, y, w, h = bounding_box
    center = ( ((x + w) // 2), ((y + h) // 2) )
    
    # Draw the countour number on the image
    cv2.putText(img=img,
                text=f"{contour_id}",
                org=center, # Bottom-left corner of the text string in the image.
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=10, 
                color=(255, 255, 255),
                thickness=40)

    return img

def XLargestBlobs(mask, top_X=None):
# Find all contours from binarised image.
    # Note: parts of the image that you want to get should be white.
    contours, hierarchy = cv2.findContours(image=mask,
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_NONE)
    
    n_contours = len(contours)
    
    # Only get largest blob if there is at least 1 contour.
    if n_contours > 0:
        
        # Make sure that the number of contours to keep is at most equal 
        # to the number of contours present in the mask.
        if n_contours < top_X or top_X == None:
            top_X = n_contours
        
        # Sort contours based on contour area.
        sorted_contours, bounding_boxes = SortContoursByArea(contours=contours,
                                                             reverse=True)
        
        # Get the top X largest contours.
        X_largest_contours = sorted_contours[0:top_X]
        
        # Create black canvas to draw contours on.
        to_draw_on = np.zeros(mask.shape, np.uint8)
        
        # Draw contours in X_largest_contours.
        X_largest_blobs = cv2.drawContours(image=to_draw_on, # Draw the contours on `to_draw_on`.
                                           contours=X_largest_contours, # List of contours to draw.
                                           contourIdx=-1, # Draw all contours in `contours`.
                                           color=1, # Draw the contours in white.
                                           thickness=-1) # Thickness of the contour lines.
        
    return n_contours, X_largest_blobs

def InvertMask(mask):

    inverted_mask = np.zeros(mask.shape, np.uint8)
    inverted_mask[mask == 0] = 1
    
    return inverted_mask

def InPaint(img, mask, flags="telea", inpaintRadius=30):

    # First convert to `img` from float64 to uint8.
    img = 255 * img
    img = img.astype(np.uint8)
    
    # Then inpaint based on flags.
    if flags == "telea":
        inpainted_img = cv2.inpaint(src=img,
                                    inpaintMask=mask,
                                    inpaintRadius=inpaintRadius,
                                    flags=cv2.INPAINT_TELEA)
    elif flags == "ns":
        inpainted_img = cv2.inpaint(src=img,
                                    inpaintMask=mask,
                                    inpaintRadius=inpaintRadius,
                                    flags=cv2.INPAINT_NS)
    
    return inpainted_img

def ApplyMask(img, mask):
    
    masked_img = img.copy()
    masked_img[mask == 0] = 0
    
    return masked_img

def HorizontalFlip(mask):
    # Get number of rows and columns in the image.
    nrows, ncols = mask.shape
    x_center = ncols // 2
    y_center = nrows // 2
    
    # Sum down each column.
    col_sum = mask.sum(axis=0)
    # Sum across each row.
    row_sum = mask.sum(axis=1)
    
    left_sum = sum(col_sum[0:x_center])
    right_sum = sum(col_sum[x_center:-1])
    top_sum = sum(row_sum[0:y_center])
    bottom_sum = sum(row_sum[y_center:-1])
    
    if left_sum < right_sum:
        horizontal_flip = True
    else:
        horizontal_flip = False
        
    return horizontal_flip

def clahe(img, clip=2.0, tile=(8, 8)):

    # Convert to uint8.
    # img = skimage.img_as_ubyte(img)
    img = cv2.normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    img_uint8 = img.astype("uint8")

    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    clahe_img = clahe_create.apply(img_uint8)

    return clahe_img

def Pad(img):

    nrows, ncols = img.shape
    
    # If padding is required...
    if nrows != ncols:
        
        # Take the longer side as the target shape.    
        if ncols < nrows:
            target_shape = (nrows, nrows)
        elif nrows < ncols:
            target_shape = (ncols, ncols)
        
        # Pad.
        padded_img = np.zeros(shape=target_shape)
        padded_img[ :nrows, :ncols ] = img
        
        return padded_img
    
    # If padding is not required, return original image.
    elif nrows == ncols:
        
        return img

def preprocess_image(image_path):# Convert BGR to RGB
    image = cv2.resize(image_path, (224, 224))  # Resize to 224x224 # Normalize pixel values between 0 and 1
    return image
# Run the above function and store its results in a variable.   
full_file_paths = get_filepaths("/Users/MayraBerrones/Documents/DDSM-CIBIS/Full")

selected_paths = []


for f in full_file_paths:
  if f.endswith(".png"):
    print(f)
    ds = cv2.imread(f)
    ds_grayscale = cv2.cvtColor(ds, cv2.COLOR_BGR2GRAY)
    name = Path(f).stem
    cropped_img = CropBorders(img=ds_grayscale)
    binarised_img = OwnGlobalBinarise(img=cropped_img, thresh=0.1, maxval=1.0)
    edited_mask = OpenMask(mask=binarised_img, ksize=(33, 33), operation="open")
    _, X_largest_blobs = XLargestBlobs(mask=edited_mask, top_X=1)
    inverted_mask = InvertMask(X_largest_blobs)
    masked_img = ApplyMask(img=cropped_img, mask=X_largest_blobs)
    horizontal_flip = HorizontalFlip(mask=X_largest_blobs)
    if horizontal_flip:
        flipped_img = np.fliplr(masked_img)
    else:
        flipped_img = masked_img
    clahe_img = clahe(img=flipped_img)
    padded_img = Pad(img=clahe_img)
    resized_image = preprocess_image(padded_img)
    save_path = os.path.join(f"/Users/MayraBerrones/Documents/DDSM-CIBIS/Full_enhance/{name}.png")
    cv2.imwrite(filename=save_path, img = resized_image)

print('Done ...')