import os
import cv2
from ultralytics import YOLO
from deepface import DeepFace

def crop_face_from_doc(doc_img):
    model = YOLO('yolov8n.pt')
    results = model(doc_img, save=False)
    x1, y1, x2, y2 = -1, -1, -1, -1
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            clsIdx = int(box.cls[0])
            if clsIdx==0:
                cropped = doc_img[y1:y2, x1:x2]

    assert x1>=0, "image not found"   
    return cropped

def live_doc_comparison(live_img, doc_img):
    cropped_doc = crop_face_from_doc(doc_img) 
    result = DeepFace.verify(live_img, cropped_doc)
    if result["verified"]:
        return True
    else:
        return False
