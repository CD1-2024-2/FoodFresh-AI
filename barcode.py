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
        for i in range(-30, 31, 5):
            _, tmp = cv2.threshold(rotate(img, a*180/math.pi), mean+i, 255, cv2.THRESH_BINARY)
            for obj in pyzbar.decode(tmp):
                return obj.data.decode('utf-8')

def detect_barcodes(img):
    results = model(img, conf=0.1, iou=0.1, agnostic_nms=True, imgsz=(640, 640))
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            res = read_barcodes(img[y1:y2, x1:x2])
            if res != None:
                return res

def barcode2name(barcode):
    bar2name = {
        "8801043015400": "(주)농심 포테토칩 올디스타코맛 50g",
        "8801043005814": "칩포테토 오리지날 60g",
        "8801062518210": "롯데웰푸드(주) 칸쵸",
        "8801111904292": "크라운제과 스낵 카라멜콘메이플 74G",
        "8801111186100": "크라운_쿠크다스화이트72g",
        "8801111113199": "크라운 와플 버터와플 52G",
        "8801111951944": "(주)크라운제과 크라운산도 딸기 61g",
        "8801043071253": "포테토칩 먹태청양마요맛 50g",
        "8801111937771": "(주)크라운제과 쟈키쟈키 70g",
        "8801111611312": "(주)크라운제과 콘칩 C콘칩 70g",
        "8801111770217": "크라운 스낵 야채타임 70G",
        "8801111180986": "(주)크라운제과 카라멜콘 땅콩 72g"
    }
    if barcode in bar2name: return bar2name[barcode]
    return "food"
