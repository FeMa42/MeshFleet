import os
import requests
import csv
import pandas as pd
import torch
from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
from setting import g_yolo_model_path
# MODEL_PATH = '../00_model_checkpoints/yolov10x.pt'

category_dict = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

class YOLO_helper:
    def __init__(self, device="cuda"):
        # torch.serialization.add_safe_globals([YOLO])
        # torch.serialization.safe_globals([YOLO])
        self.model = YOLO(g_yolo_model_path).to(device)
        self.model.eval()

    def label_single_image(self, pil_img):
        cv2_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) 
        # scale image to 640 on the longest side
        scale = 640 / max(cv2_image.shape[:2])
        cv2_image = cv2.resize(cv2_image, (0, 0), fx=scale, fy=scale)
        results = self.model(source=cv2_image, conf=0.25, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        labels = [
            f"{category_dict[class_id]}:{confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        return labels

    def label_batch_images(self, pil_images):
        cv2_images = [cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) for pil_img in pil_images]
        results = self.model(source=cv2_images, conf=0.25, verbose=False)
        labels = []
        for result in results:
            detections = sv.Detections.from_ultralytics(result)
            labels_single_image = [
                f"{category_dict[class_id]}:{confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
                ]
            internal_labels = ""
            for label in labels_single_image:
                if len(internal_labels) > 0:
                    internal_labels += ", "
                internal_labels += label #.split(":")[0]
            if len(internal_labels) == 0:
                internal_labels = " "
            labels.append(internal_labels)
        return labels