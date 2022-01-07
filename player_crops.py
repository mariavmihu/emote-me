from PIL import Image
from enum import IntEnum

class Pos(IntEnum):
    LEFT = 0
    TOP = 1
    WIDTH = 2
    HEIGHT = 3

def crop_chibi_face(bbox, frames_base,save_path,frame,i):
    image = Image.open(f'{frames_base}/{frame}.jpg')

    #PIL's crop requires a tuple of box=(top_left_X, top_left_Y, bottom_right_X, bottom_right_Y)
    crop_box = (bbox[Pos.LEFT], bbox[Pos.TOP], bbox[Pos.LEFT]+bbox[Pos.WIDTH], bbox[Pos.TOP]+bbox[Pos.HEIGHT])
    image.crop(crop_box).save(f'{save_path}/{frame}-{i}.jpg', quality=100)
