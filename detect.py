import torch
import types
import datetime
from io import BytesIO
from PIL import Image

from classify import classify
from barcode import detect_barcodes, barcode2name
from date import read_date
from img_utils import add_padding, image_to_base64

from ultralytics import YOLO
from ultralytics.utils import ops
from ultralytics.engine.results import Results

import base64

def postprocess(self, preds, img, orig_imgs):
    for i in range(preds[0].shape[0]):
        for j in range(preds[0].shape[2]):
            preds[0][i,4,j] = sum(preds[0][i,4:,j])
            preds[0][i,5:,j] = torch.Tensor([0 for k in range(preds[0].shape[1]-5)])
    preds = ops.non_max_suppression(
        preds,
        self.args.conf,
        self.args.iou,
        agnostic=self.args.agnostic_nms,
        max_det=self.args.max_det,
        classes=self.args.classes,
    )

    if not isinstance(orig_imgs, list):
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

    results = []
    for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
    return results

model = YOLO('yolo11n.pt')
model()
model.predictor.postprocess = types.MethodType(postprocess, model.predictor)

def detect_object(img):
    ret = []
    img = add_padding(img)
    result = model(img, conf=0.1, iou=0.1, half=True, imgsz=((1280, 1280)))[0]
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = img.shape[0:2]
        rect = img[y1:y2, x1:x2]
        barcode = detect_barcodes(rect)
        if barcode != None:
            ret.append({
                "rect": [x1/w, y1/h, (x2-x1)/w, (y2-y1)/h],
                "tag": barcode2name(barcode),
                "barcode": barcode,
                "date": read_date(rect)
            })
        else:
            tag, date = classify(rect)
            ret.append({
                "rect": [x1/w, y1/h, (x2-x1)/w, (y2-y1)/h],
                "tag": tag,
                "barcode": "",
                "date": (datetime.datetime.now()+datetime.timedelta(days=date)).strftime('%Y-%m-%d')
            })
    return ret
