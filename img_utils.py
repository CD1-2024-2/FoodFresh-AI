import numpy as np
import cv2
import math

def get_angle(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 100, 200)

    h, w = img.shape
    thres = (h+w)//4
    lines = cv2.HoughLinesP(img, 1, np.pi/180, thres, None, 20, 2)
    if lines is None:
        return [x/180*math.pi for x in range(0, 360, 30)]

    cand1, cand2 = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2-x1, y2-y1
        if dx < 0: dx, dy = -dx, -dy

        if abs(dy) <= abs(dx):
            angle = math.atan(dy/dx)
            cand1.append(angle)
        else:
            angle = math.atan(dx/dy)
            cand2.append(angle)
    ret = []
    if len(cand1) > 0:
        ret.append(sum(cand1)/len(cand1))
        ret.append(sum(cand1)/len(cand1)+math.pi/2)
        ret.append(sum(cand1)/len(cand1)+math.pi)
        ret.append(sum(cand1)/len(cand1)+math.pi*3/2)
    if len(cand2) > 0:
        ret.append(-sum(cand2)/len(cand2))
        ret.append(-sum(cand2)/len(cand2)+math.pi/2)
        ret.append(-sum(cand2)/len(cand2)+math.pi)
        ret.append(-sum(cand2)/len(cand2)+math.pi*3/2)
    ret = list(set(map(lambda x: int(x/math.pi*180+360)%360,ret)))
    return ret

def rotate(img, angle):
    h, w = img.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def add_padding(img):
    h, w = img.shape[:2]
    diff = abs(h-w)
    if h > w:
        return cv2.copyMakeBorder(img, 0, 0, diff//2, (diff+1)//2, cv2.BORDER_CONSTANT, value=[0])
    elif w > h:
        return cv2.copyMakeBorder(img, diff//2, (diff+1)//2, 0, 0, cv2.BORDER_CONSTANT, value=[0])
    return img
