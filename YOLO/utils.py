import torch
import json, os, config
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont
import torchvision.transforms as T
import matplotlib.patches as patches


def get_iou(boxA, boxB, epsilon=1e-5):
    #coordinte of intersection box:
    x1=max(boxA[0], boxB[0]) #left-most value
    y1=max(boxA[1], boxB[1]) #top-most value
    x2=min(boxA[2], boxB[2]) #right-most value
    y2=min(boxA[3], boxB[3]) #bottom-most value

    width=(x2-x1)
    height=(y2-y1)

    if (width < 0) or (height <0):
        return 0
    area_overlap=width*height

    # calculating combined area:
    area_a=(boxA[2]-boxA[0]) * (boxA[3] - boxA[1])
    area_b=(boxB[2]-boxB[0]) * (boxB[3] - boxB[1])
    area_combined=area_a + area_b - area_overlap

    iou=area_overlap / (area_combined + epsilon) #epsilon  to avoid undetermined value
    return iou


def load_class_dict():
    if os.path.exists(config.CLASSES_PATH):
        with open(config.CLASSES_PATH, 'r') as file:
            return json.load(file)
        
    new_dict={}
    save_class_dict(new_dict)
    return new_dict


def save_class_dict(obj):
    folder=os.path.dirname(config.CLASSES_PATH)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(config.CLASSES_PATH, 'w') as file:
        json.dump(obj, file, indent=2)



def scale_bbox_coord(coord, center, scale):
    return ((coord-center)*scale)+center

def bbox_attr(data, i):
    """returns the ith attribute of each bounding box in data"""
    attr_start=config.C + i
    return data[..., attr_start::5]