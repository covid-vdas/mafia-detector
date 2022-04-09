ONNX_PATH = "library/Detector/weights/best-0312-5.onnx"
CLASS_PATH = "library/Detector/weights/vdas.names"
YOLO_MODEL = '../Yolov5_DeepSort_Pytorch/weights/best-0317-13.pt'
DEEP_SORT_MODEL = 'osnet_ibn_x1_0_MSMT17'

VDAS_YAML_PATH = '../Yolov5_DeepSort_Pytorch/yolov5/models/hub/vdas.yaml'
DEEP_SORT_YAML_PATH = '../Yolov5_DeepSort_Pytorch/deep_sort/configs/deep_sort.yaml'

DATA_NAME = "Oxford_Town_Centre"
DATA_PATH = "dataset/" + DATA_NAME + ".mp4"
IMG_PATH = "dataset/" + DATA_NAME + ".png"

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
 
MIN_CONFIDENCE_RATE = 0.4
NMS_THRESH = 0.4
MIN_DISTANCE = 200
OBJECT_DISTANCE = 400

# Colors
COLOR = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
