import numpy as np
import cv2
import random
import math

from scipy.interpolate import UnivariateSpline


'''
def perspective_transform(im, direction="random", amount=None);

def warm_img(img):

def compress_jpeg(img, quality):

def motion_blur(img, size=9);

'''

def perspective_transform(im, direction="random", amount=None):
    '''
    Performs a perspective transform on an image. Doesn't 
    crop anything since our background is transparent.
    
    Follows same set of transforms used in the Augmentor library,
    but theirs crops the image, cutting off our beautiful traffic signs
    '''
    directions = ["left", "right", "forward", "backward"] + ["skew_%d" % x for x in range(8)]
    if direction == "random":
        direction = random.choice(directions)
    
    # image dimensions
    h, w = im.shape[:2]
    
    # starting points of image
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])  
    
    
    axis = min(w, h)
    amt = axis * (random.random()*0.4 if amount is None else amount)

    
    if direction == "left":
        pts2 = np.float32([[0, 0+amt], [w, 0], [0, h-amt], [w, h]])
    elif direction == "right":
        pts2 = np.float32([[0, 0], [w, 0+amt], [0, h], [w, h-amt]])
    elif direction == "forward":
        pts2 = np.float32([[0, 0], [w, 0], [0+amt, h], [w-amt, h]])
    elif direction == "backward":
        pts2 = np.float32([[amt, amt], [w-amt, amt], [0, h], [w, h]])

    # Shift top left corner to the left
    elif direction == "skew_0":
        pts2 = np.float32([[0, 0], [w, 0], [0+amt, h], [w, h]])

    # Shift top left corner up
    elif direction == "skew_1":
        pts2 = np.float32([[0, 0], [w, 0+amt], [0, h], [w, h]])

    # Shift top right corner to the right
    elif direction == "skew_2":
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w-amt, h]])

    # Shift top right corner up
    elif direction == "skew_3":
        pts2 = np.float32([[0, 0+amt], [w, 0], [0, h], [w, h]])  

    # Shift bottom right corner right
    elif direction == "skew_4":
        pts2 = np.float32([[0, 0], [w-amt, 0], [0, h], [w, h]])  

    # Shift bottom right corner down
    elif direction == "skew_5":
        pts2 = np.float32([[0, 0], [w, 0], [0, h-amt], [w, h]])  

    # Shift bottom left corner left
    elif direction == "skew_6":
        pts2 = np.float32([[0+amt, 0], [w, 0], [0, h], [w, h]])  
    
    # Shift bottom left corner down
    elif direction == "skew_7":
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h-amt]])  
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(im, matrix, (w, h))
    x_min = int(min([pt[0] for pt in pts2]))
    x_max = int(max([pt[0] for pt in pts2]))
    y_min = int(min([pt[1] for pt in pts2]))
    y_max = int(max([pt[1] for pt in pts2]))

    return result[y_min:y_max, x_min:x_max]



     
def create_LUT_8UC1(x, y):
    spl = UnivariateSpline(x, y)
    return spl(range(256))

def warm_img(img):
    '''
    Code taken from Michael Beyeler:
    http://www.askaswiss.com/2016/02/how-to-manipulate-color-temperature-opencv-python.html
    '''
    incr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
        [0, 70, 140, 210, 256])
    decr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
        [0, 30, 80, 120, 192])

    img_bgr_in = img

    c_b, c_g, c_r = cv2.split(img_bgr_in)
    c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
    img_bgr_warm = cv2.merge((c_b, c_g, c_r))

    c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)

    # increase color saturation
    c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_bgr_warm,
        cv2.COLOR_BGR2HSV))
    c_s = cv2.LUT(c_s, incr_ch_lut).astype(np.uint8)

    img_bgr_warm = cv2.cvtColor(cv2.merge(
        (c_h, c_s, c_v)),
        cv2.COLOR_HSV2BGR)
    return img_bgr_warm
#     cv2.imwrite(\"warm.png\", img_bgr_warm)
    # plt.imshow(img_bgr_wa


def compress_jpeg(img, min_quality, max_quality):
    quality = random.randint(min_quality, max_quality)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    result = cv2.imdecode(encimg, 1)
    return result

def motion_blur(img, size=9):
    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[(size-1)//2, :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    # applying the kernel to the input image
    output = cv2.filter2D(img, -1, kernel_motion_blur)
    return output


def paste(background, obj_rgba, x=0, y=0, blur_mode="gaussian"):
    '''
    Pastes a rgba image onto a background image, blending 
    the border and preserving transparency
    '''
    if obj_rgba is None or background is None:
        print("Error: Could not read images")
        return
    
    if obj_rgba.shape[2] != 4:
        print("Error: Object has no alpha channel")
        return
    
    bg_h, bg_w = background.shape[:2]
    obj_h, obj_w = obj_rgba.shape[:2]
    if (x+obj_w) > bg_w or (y+obj_h) > bg_h:
        print("Error: object exceeds image bounds")
        return
    
    # Split object into rgb and alpha channels
    obj_channels = cv2.split(obj_rgba)
    
    alpha = obj_channels[-1]
    obj_rgb = cv2.merge(obj_channels[:-1])
    
    alpha = cv2.merge((alpha, alpha, alpha))
    
    # Convert object and background to float arrays
    obj_rgb = obj_rgb.astype(float)
    background = background.astype(float)
    
    # Blur the alpha filter to create the blending effect
    if blur_mode == "gaussian":
        alpha = cv2.GaussianBlur(src=alpha, ksize=(5, 5), sigmaX=2)
    
    alpha = alpha.astype(float)/255  
    
    # Combine mask and object
    obj_rgb = cv2.multiply(alpha, obj_rgb)
  
    # Paste object on background
    background[y:y+obj_h, x:x+obj_w] = cv2.multiply(1.0 - alpha, background[y:y+obj_h, x:x+obj_w])
    background[y:y+obj_h, x:x+obj_w] = cv2.add(obj_rgb, background[y:y+obj_h, x:x+obj_w])
    
    return background

def isOverlapping1D(box1, box2):
    '''
    Checks if two 1D boxes (intervals) are overlapping
    https://stackoverflow.com/questions/20925818/algorithm-to-check-if-two-boxes-overlap
    '''
    xmin1, xmax1 = min(box1), max(box1)
    xmin2, xmax2 = min(box2), max(box2)
    return xmax1 >= xmin2 and xmax2 >= xmin1

def isOverlapping2D(box1, box2):
    '''
    Checks if two 2D boxes are overlapping
    https://stackoverflow.com/questions/20925818/algorithm-to-check-if-two-boxes-overlap
    '''
    box1_x = (box1[0][0], box1[1][0])
    box2_x = (box2[0][0], box2[1][0])
    xmin1, xmax1 = min(box1_x), max(box1_x)
    xmin2, xmax2 = min(box2_x), max(box2_x)

    box1_y = (box1[0][1], box1[1][1])
    box2_y = (box2[0][1], box2[1][1])
    ymin1, ymax1 = min(box1_y), max(box1_y)
    ymin2, ymax2 = min(box2_y), max(box2_y)

    return isOverlapping1D((xmin1, xmax1), (xmin2, xmax2)) and isOverlapping1D((ymin1, ymax1), (ymin2, ymax2))


def paste_random(background, obj, ignore=None):
    '''
    Pastes an object randomly onto a background and 
    returns the modified image as well as the bounding 
    boxes
    '''
    if obj is None or background is None:
        print("Error: Could not read images")
        return
    
    # Get background and object dimensions
    bg_h, bg_w = background.shape[:2]
    obj_h, obj_w = obj.shape[:2]
    
    # Get area of background and object
    bg_area = bg_h * bg_w
    obj_area = obj_h * obj_w

    # Resize the object to a random percentage of the background area
    obj_resize_scale = np.random.uniform(0.001, 0.05)
    w_h_ratio = math.sqrt((obj_resize_scale * bg_area) / obj_area)
    new_obj_h, new_obj_w = obj_h * w_h_ratio, obj_w * w_h_ratio
    
    # change object height and width randomly
    new_obj_h = int(new_obj_h * np.random.normal(loc=1, scale=0.1))
    new_obj_w = int(new_obj_w * np.random.normal(loc=1, scale=0.1))
    
    # Resize object
    obj = cv2.resize(obj, dsize=(new_obj_w, new_obj_h), interpolation = cv2.INTER_CUBIC)
    
    # Pick random spot to paste image
    good_spots = np.ones((bg_w, bg_h))

    x, y = random.randint(0, bg_w-new_obj_w), random.randint(0, bg_h-new_obj_h)

    for _ in range(1000):
        no_overlap = True
        obj_bbox = ((x, y), (x+new_obj_w, y+new_obj_h))
        for ignore_bbox in ignore:
            if(isOverlapping2D(obj_bbox, ignore_bbox)):
                no_overlap = False
        if no_overlap:
            break
        else:
            x, y = random.randint(0, bg_w-new_obj_w), random.randint(0, bg_h-new_obj_h)
            

    
    # Paste the image
    result = paste(background, obj, x, y)
    
    # Get bounding boxes
    bbox = ((x, y), (x+new_obj_w, y+new_obj_h))
    
    
    return result, bbox
    