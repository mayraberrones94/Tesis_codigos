import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageChops
import numpy as np
import cv2
import os



if __name__ == "__main__":
    for f in os.listdir('.'):
        if f.endswith('.png'):
            im = Image.open(f)
            fn, fext = os.path.splitext(f)
            img_draw = ImageDraw.Draw(im)
            img_draw.rectangle((0, 0, 350, 95),  fill='black')
            img_draw.rectangle((1024, 0, 550, 95),  fill='black')
            try:
                im.save('{}.png'.format(fn))
            except AttributeError:
                print("Not found {}".format(img))
