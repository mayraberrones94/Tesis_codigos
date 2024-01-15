import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageChops
import numpy as np
import cv2

import os



if __name__ == "__main__":
    for f in os.listdir('.'):
        if f.endswith('.png'):
            img = cv2.imread(f)
            fn, fext = os.path.splitext(f)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = image > 0
            coords = np.argwhere(mask)
            x0, y0 = coords.min(axis=0)
            x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
            cropped = image[x0:x1, y0:y1]
            result = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
            try:
                cv2.imwrite('crop/{}.png'.format(fn), result)
            except AttributeError:
                print("Not found {}".format(img))

