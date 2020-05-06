#-------------------------------------------------------------------------------
# Name:        Bounding Box to YOLO Conversion script
# Purpose:     Converts bounding box labels from <x1, y1, x2, y2, w, h> format 
#               to <x_ratio, y_ratio, w_ratio, h_ratio> for YOLO training
# Author:      Andrew Hu
# Created:     03/12/2019
#
#-------------------------------------------------------------------------------


from PIL import Image
import os
from shutil import copyfile

OUTPUT_DIR = "YOLO_Formatted/"

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


with open("classes.txt") as f:
    classes = [x.strip() for x in f.readlines()]
print("Classes:", classes)


obj_paths = [x[:-4] for x in os.listdir("generated") if os.path.splitext(x)[1] == ".jpg"]

for obj_path in obj_paths:
    im_path = obj_path + ".jpg"
    ann_path = obj_path + ".txt"
    copyfile("generated/"+im_path, OUTPUT_DIR + im_path)
    with open("generated/" + ann_path) as input_file:
        labels = [x.strip() for x in input_file.readlines()]
    formatted_file = open(os.path.join(OUTPUT_DIR, ann_path), "w")
    for label in labels:
        label = label.split()
        x1, y1, x2, y2, w, h = tuple([int(i) for i in label[:-1]])
        center_x = float(x1+x2)/2
        center_y = float(y1+y2)/2
        center_x_f = float(center_x)/int(w)
        center_y_f = float(center_y)/int(h)
        obj_width_x = float(x2-x1)/int(w)
        obj_height_y = float(y2-y1)/int(h)
        obj_class = classes.index(label[6])
        formatted_file.write("%d %f %f %f %f\n"%(obj_class, center_x_f, center_y_f, obj_width_x, obj_height_y)) 
    formatted_file.close()

print("Done!")








# folders = [x for x in os.listdir("Images") if x != "examples"]

# with open("classes.txt") as f:
#     classes = [x.strip() for x in f.readlines()]
# print("Classes:", classes)

# for folder in folders:
#     print("Now converting class\"", folder, "\"")
#     # only convert labels from the images we keep
#     labels = [x[:-4] for x in os.listdir(os.path.join("Images", folder)) if x.endswith(".jpg")]
#     if not os.path.exists(os.path.join(OUTPUT_DIR, folder)):
#         os.mkdir(os.path.join(OUTPUT_DIR, folder))

#     for label_name in labels:
#         copyfile(os.path.join("Images", folder, label_name+".jpg"), os.path.join(OUTPUT_DIR, folder, label_name+".jpg"))
#         with open(os.path.join("Labels", folder, label_name+".txt")) as input_file:
#             labels = [x.strip() for x in input_file.readlines()]
#         formatted_file = open(os.path.join(OUTPUT_DIR, folder, label_name+".txt"), "w")
#         for label in labels:
#             label = label.split()
#             x1, y1, x2, y2, w, h = tuple([int(i) for i in label[:-1]])
#             center_x = float(x1+x2)/2
#             center_y = float(y1+y2)/2
#             center_x_f = float(center_x)/int(w)
#             center_y_f = float(center_y)/int(h)
#             obj_width_x = float(x2-x1)/int(w)
#             obj_height_y = float(y2-y1)/int(h)
#             obj_class = classes.index(label[6])
#             formatted_file.write("%d %f %f %f %f\n"%(obj_class, center_x_f, center_y_f, obj_width_x, obj_height_y)) 
#         formatted_file.close()
#     # print("done")
