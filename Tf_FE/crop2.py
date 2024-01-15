import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageChops
import numpy as np
import cv2


im = Image.open('/Users/mayraberrones/Documents/Anomalia/imagenprueba.png')
img_draw = ImageDraw.Draw(im)
img_draw.rectangle((0, 0, 350, 95),  fill='black')
img_draw.rectangle((1024, 0, 550, 95),  fill='black')

im.save('crop/image_flip.png')


img = cv2.imread('/Users/mayraberrones/Documents/Anomalia/crop/image_flip.png')

image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Mask of non-black pixels (assuming image has a single channel).
mask = image > 0

# Coordinates of non-black pixels.
coords = np.argwhere(mask)

# Bounding box of non-black pixels.
x0, y0 = coords.min(axis=0)
x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top

# Get the contents of the bounding box.
cropped = image[x0:x1, y0:y1]
result = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)

cv2.imwrite('crop/imglol.png', result)