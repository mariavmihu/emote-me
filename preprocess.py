import pandas as pd
import numpy as np
import os
from PIL import Image
from enum import IntEnum
import sys

BASE_DIR = os.getcwd()
_SAVE_PATH = f'{BASE_DIR}/data/cropped_images'

class Pos(IntEnum):
    LEFT = 0
    TOP = 1
    WIDTH = 2
    HEIGHT = 3

def crop_chibi_face(bbox, image_path, save_path, frame_name):
    
    def convert_yolo_bbox(img_w, img_h, box):
        
        # box: xc, yc, w, h 
        xc, yc = int(np.round(box[0]*img_w)), int(np.round(box[1]*img_h))
        w, h = int(np.round(box[2]*img_w)), int(np.round(box[3]*img_h))

        left = xc - int(np.round(w/2))
        top = yc - int(np.round(h/2))

        return [left, top, w, h]
    
    image = Image.open(image_path)
    image_width,image_height = image.size
    
    bbox = convert_yolo_bbox(image_width,image_height, bbox)
    
    try:
        #PIL's crop requires a tuple of box=(top_left_X, top_left_Y, bottom_right_X, bottom_right_Y)
        crop_box = (bbox[Pos.LEFT], bbox[Pos.TOP], bbox[Pos.LEFT]+bbox[Pos.WIDTH], bbox[Pos.TOP]+bbox[Pos.HEIGHT])
        image.crop(crop_box).save(f'{save_path}/{frame_name}.jpeg', quality=100)
    except Exception as e:
        print(e)

    return

def load_data_and_crop(labels_path, image_name=''):
    chibi_faces = pd.read_fwf(labels_path, header=None)
    chibi_faces.columns = ["label", "x_center", "y_center", "width", "height"]
    
    for i, (label, x_center, y_center, width, height) in chibi_faces[["label", "x_center", "y_center", "width", "height"]].iterrows():
        image_path = f'{labels_path[:-4]}.jpeg'
        if image_name == '':
            image_name = os.path.basename(labels_path)[:-4]
        crop_chibi_face([x_center, y_center, width, height], image_path, _SAVE_PATH, f'{image_name}_{i}')
    
    return

def begin_cropping():
    
    _BASE_IMAGE_DIR = 'C:\\Users\\mvmih\\OneDrive\\Pictures\\bens_stickers\\temp'
    
    '''for file in os.listdir(f'{_BASE_IMAGE_DIR}\\outliers'):
        if file.endswith('.txt') and file != "classes.txt":
            load_data_and_crop(f'{_BASE_IMAGE_DIR}\\outliers\\{file}')
    '''
    
    for folder in os.listdir(_BASE_IMAGE_DIR):
        if folder == "outliers":
            continue
        
        for file in os.listdir(f'{_BASE_IMAGE_DIR}/{folder}'):
            if file == 'classes.txt':
                continue
        
            if file.endswith('.txt'):
                labels_path = f'{_BASE_IMAGE_DIR}/{folder}/{file}'
        
        for file in os.listdir(f'{_BASE_IMAGE_DIR}/{folder}'):
            print(file)
            try:
                load_data_and_crop(labels_path, image_name=file[:-4])
            except Exception as e:
                print(e)

    return

if __name__ == "__main__":
    print("beginning")
    begin_cropping()
    print("done")