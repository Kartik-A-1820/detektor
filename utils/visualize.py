import cv2
import numpy as np

def draw_boxes(img, boxes):
    for box in boxes:
        x1,y1,x2,y2 = map(int, box)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    return img
