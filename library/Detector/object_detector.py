import cv2
from library.Detector.object_detection import ObjectDetection
from library.Detector.object_config import *
from mafiaDetector.settings import *

def detect_from_img(img_path):
    # Initialize Object Detection
    od = ObjectDetection()
    # Create image object
    img = cv2.imread(str(DETECT_ROOT / img_path))
    # Format img suitable with input from yolov5(3 chanels of color)
    ref_img = od.format_yolov5(img)
    # Get output predictions
    outs = od.detect(ref_img)
    # Detect person in frame
    class_ids, confidences, boxes = od.wrap_detection(ref_img, outs[0], is_person=False)

    # Return all bounding box related to person, mask
    # and draw all bounding box on img
    bounding_box = []
    for (i, (box)) in enumerate(boxes):
        (startX, startY, width, height) = box
        cv2.rectangle(img, box, GREEN, 2)
        cv2.putText(img, str(od.classes[class_ids[i]]), (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, .5, RED)
        bounding_box.append([class_ids[i], box])
    cv2.imwrite(str(DETECTED_ROOT / img_path), img)

    return bounding_box
