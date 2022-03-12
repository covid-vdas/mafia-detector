import cv2
import sys
import numpy as np
from library.Detector.object_config import *

class ObjectDetection:
    def __init__(self):
        # Load Network
        is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"
        self.net = self.build_model(is_cuda)
        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(80, 3))

    def load_class_names(self, classes_path=CLASS_PATH): 
        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                # if class_name == 'person':
                self.classes.append(class_name)

        self.colors = np.random.uniform(0, 255, size=(80, 3))
        return self.classes

    def build_model(self, is_cuda, onnx_path = ONNX_PATH):
        net = cv2.dnn.readNet(onnx_path)
        if is_cuda:
            print("Attempty to use CUDA")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            print("Running on CPU")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def format_yolov5(self, frame):
        row, col, _ = frame.shape
        # print(frame.shape)
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    def detect(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
        self.net.setInput(blob)
        predictions = self.net.forward()
        return predictions

    def wrap_detection(self, frame, output_data, is_person=False):
        class_ids = []
        confidences = []
        boxes = [] 

        rows = output_data.shape[0]

        frame_width, frame_height, _ = frame.shape

        x_factor = frame_width / INPUT_WIDTH
        y_factor =  frame_height / INPUT_HEIGHT
     
        for r in range(rows):
            row = output_data[r]
            confidence = row[4]
            if confidence >= MIN_CONFIDENCE_RATE:
                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if is_person:
                    if (classes_scores[class_id] > 0.25 and class_id == self.classes.index("person")):

                        confidences.append(confidence) 
                        class_ids.append(class_id)

                        x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                        left = int((x - 0.5 * w) * x_factor)
                        top = int((y - 0.5 * h) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        box = np.array([left, top, width, height])
                        boxes.append(box) 
                else:  
                    if (classes_scores[class_id] > .25 ):

                        confidences.append(confidence) 
                        class_ids.append(class_id)

                        x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                        left = int((x - 0.5 * w) * x_factor)
                        top = int((y - 0.5 * h) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        box = np.array([left, top, width, height])
                        boxes.append(box)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

        result_class_ids = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])

        return result_class_ids, result_confidences, result_boxes 
 
