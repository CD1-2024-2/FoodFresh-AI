import math
import cv2
import numpy as np
from pyzbar import pyzbar
from ultralytics import YOLO

from img_utils import rotate, get_angle, add_padding

model = YOLO('models/barcode_detect.pt')

def read_barcodes(img):
    for obj in pyzbar.decode(img):
        return obj.data.decode('utf-8')
    h, w, _ = img.shape
    angles = get_angle(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.int_(np.mean(img))
    img = add_padding(img)
    for a in angles:
        for i in range(-20, 21, 5):
            _, tmp = cv2.threshold(rotate(img, a*180/math.pi), mean+i, 255, cv2.THRESH_BINARY)
            for obj in pyzbar.decode(tmp):
                return obj.data.decode('utf-8')

def detect_barcodes(img):
    results = model(img, conf=0.3, iou=0.1, agnostic_nms=True, imgsz=img.shape[0:2])
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            res = read_barcodes(img[y1:y2, x1:x2])
            if res != None:
                return res
