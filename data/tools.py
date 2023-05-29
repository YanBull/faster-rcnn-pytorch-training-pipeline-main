import cv2 as cv
import os

def find_smallest_image(dir=os.PathLike):
    
    min_h = 100000
    min_w = 100000
    
    images = [os.path.join(dir, x) for x in os.listdir(dir) if x[-3:] == "jpg"]
    for image in images:
        shape = cv.imread(image).shape
        
        if shape[0] < min_h:
            min_h = shape[0]
        
        if shape[1] < min_w:
            min_w = shape[1]
    
    print("Minimal Heigth: ", min_h)
    print("Minimal Width: ", min_w)

    return min_h, min_w 