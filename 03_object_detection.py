import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import json






#This is the object detection approach it processes all images in the Datenset folder 



MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1

FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red



model_path = 'efficientdet_lite2.tflite'
image_folder_path = 'Datenset'
output_path = 'Object_Detection_output'

def detection_to_dict(detection):
    return {
        'bounding_box': {
            'x1': detection.bounding_box.origin_x,
            'y1': detection.bounding_box.origin_y,
            'x2': detection.bounding_box.width + detection.bounding_box.origin_x,
            'y2': detection.bounding_box.height + detection.bounding_box.origin_y,
        },
        'categories': [{
            'category_name': category.category_name,
            'score': category.score 
        } for category in detection.categories]
    }

def detect_objects(image_path):
    mp_image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(mp_image)
    serializable_detections = []
    for detection in detection_result.detections:
        serializable_detections.append(detection_to_dict(detection))
    return serializable_detections


def construct_bbox(results):
    constructed_bbox = {}
    for image in results:
        x1 = float("inf")
        y1 = float("inf")
        x2 = 0
        y2 = 0
        for detection in results[image]:
            x1 = min(detection["bounding_box"]["x1"],x1)
            y1 = min(detection["bounding_box"]["y1"],y1)
            x2 = max(detection["bounding_box"]["x2"],x2)
            y2 = max(detection["bounding_box"]["y2"],y2)

        constructed_bbox[image] = [x1,y1,x2,y2]
        for var in constructed_bbox[image]:
            print(var)
            if var == float("inf"):
                
                constructed_bbox[image] = []
    return constructed_bbox


thresholds = [0.3,0.6,0.8]

for threshold in thresholds:
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=threshold, max_results = 15)
    detector = vision.ObjectDetector.create_from_options(options)

    results = {}

    for image_name in os.listdir(image_folder_path):
        image_path = image_folder_path + "\\" + image_name
        results[image_name] = detect_objects(image_path)

    with open(output_path + "\\" + str(threshold) + "raw_detections.json", "w") as json_file:
        json.dump(results, json_file, indent=4)
    constructed_bbox = construct_bbox(results)
    with open(output_path + "\\" + str(threshold) + "constructedbbox.json", "w") as json_file:
        json.dump(constructed_bbox, json_file, indent=4)