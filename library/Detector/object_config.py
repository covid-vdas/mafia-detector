ONNX_PATH = "library/Detector/weights/best-0305-5.onnx"
CLASS_PATH = "library/Detector/weights/vdas.names"

DATA_NAME = "Oxford_Town_Centre"
DATA_PATH = "dataset/" + DATA_NAME + ".mp4"
IMG_PATH = "dataset/" + DATA_NAME + ".png"


INPUT_WIDTH = 640
INPUT_HEIGHT = 640

 
MIN_CONFIDENCE_RATE = 0.3
NMS_THRESH = 0.3 
MIN_DISTANCE = 200
OBJECT_DISTANCE = 400

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
