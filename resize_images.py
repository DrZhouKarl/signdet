#-------------------------------------------------------------------------------
# Name:        Image resize script
# Purpose:     Resizes images to specified height
# Author:      Andrew Hu
# Created:     03/12/2019
#
#-------------------------------------------------------------------------------


# Resizes images over specified pixel height so the images fit on screen
import os
from PIL import Image

IMAGE_DIR = "thumbnails/stop_sign"

images = [x.strip() for x in os.listdir(IMAGE_DIR)]
MAX_HEIGHT = 300 # max height in pixels

for img_path in images:
    im = Image.open(os.path.join(IMAGE_DIR, img_path))
    w, h = im.size
    if True:
        print("Resizing", img_path)
        im = im.convert('RGB') # Get rid of alpha channels
        # resize_ratio = float(MAX_HEIGHT)/h
        # new_w = resize_ratio * w
        # im = im.resize((int(new_w), MAX_HEIGHT), Image.ANTIALIAS)
        im = im.resize((MAX_HEIGHT, MAX_HEIGHT), Image.ANTIALIAS)
        im.save(os.path.join(IMAGE_DIR, img_path))


