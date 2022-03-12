import cv2
from library.Detector.object_detection import ObjectDetection
from library.Detector.object_config import *
from mafiaDetector.settings import *
import numpy as np
from math import sqrt
from scipy.spatial import distance as dist
import imutils


def detect_from_img(img_path):
    # Initialize Object Detection
    od = ObjectDetection()
    # Create image object
    img = cv2.imread(str(DETECT_ROOT / img_path))
    # Format img suitable with input from yolov5(3 chanels of color)
    img = od.format_yolov5(img)
    # Get output predictions
    outs = od.detect(img)
    # Detect person in frame
    class_ids, confidences, boxes = od.wrap_detection(img, outs[0], is_person=False)

    # Return all bounding box related to person, mask
    # and draw all bounding box on img
    bounding_box = []
    for (i, (box)) in enumerate(boxes):
        (startX, startY, width, height) = box
        cv2.rectangle(img, box, GREEN, 2)
        cv2.putText(img, str(od.classes[class_ids[i]]), (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, .5, RED)
        bounding_box.append([class_ids[i], box])
    cv2.imwrite(str(DETECTED_ROOT / img_path), img)

    return list(bounding_box)


def detect_from_video(video_path, img_path, obj_distance):
    # Initialize Object Detection
    od = ObjectDetection()
    # Create video object
    ref_video = cv2.VideoCapture(video_path)

    # Ensure that in img have exactly two person
    img_path = str(DETECT_ROOT / img_path)
    ref_image = cv2.imread(img_path)
    # Scale width, height of img suitable with video
    ref_image = scale_img_as_video(ref_image, ref_video)

    # Get distance of two person in pixel
    # Besides, we also know real distance of two person in cm(param: obj_distance)
    # So, we can calculate distance of any two person in frame
    ref_img_width = detect_distance(ref_image, od)
    print(ref_img_width)
    cv2.imwrite(str(DETECTED_ROOT / img_path), ref_image)

    # Create writer to save detected video to folder
    writer = None
    bounding_box = []

    # Loop each frame from video, and detect to get violations
    while True:
        _, frame = ref_video.read()
        if frame is None:
            print("End of video")
            break

        # Format video to 3 chanels of color
        inputImage = od.format_yolov5(frame)
        # Get output predictions after detect throw onnx file
        outs = od.detect(inputImage)
        # Only get bounding box that pass requirements
        class_ids, confidences, boxes = od.wrap_detection(inputImage, outs[0], is_person=True)

        # Get all centroids of all bounding boxes
        cent = np.array([(int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in boxes])
        if len(cent) != 0:
            D = dist.cdist(cent, cent, metric="euclidean")

            violate = set()
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # check to see if the distance between any two
                    # centroid pairs is less than the configured number
                    # of centimeter

                    # ref_img_width pixel -> KNOW_WIDTH cm
                    # D[i, j] pixel -> 200 cm (social distance)
                    # <=>
                    # 138 pixel -> 500 cm
                    # D[i, j] pixel -> 200 cm
                    # distance (pixel) = D[i, j] * KNOW_WIDTH / ref_img_width
                    if D[i, j] < MIN_DISTANCE * ref_img_width / float(obj_distance):
                        (x1, y1, w1, h1) = boxes[i]
                        (x2, y2, w2, h2) = boxes[j]
                        cent1 = [int(x1 + w1 / 2), int(y1 + h1 / 2)]
                        cent2 = [int(x2 + w2 / 2), int(y2 + h2 / 2)]

                        cv2.line(frame, cent1, cent2, RED, 2)
                        #calulate distance in meter
                        distance = D[i, j] * float(obj_distance) / ref_img_width / 100
                        distance = round(distance, 1)

                        cv2.putText(frame, str(distance), (cent1[0], cent1[1] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, RED)
                        violate.add(i)
                        violate.add(j)

        for (i, (bbox)) in enumerate(boxes):
            # extract the bounding box and centroid coordinates, then
            # initialize the color of the annotation
            (startX, startY, width, height) = bbox
            (cX, cY) = cent[i]
            color = GREEN

            # if the index pair exists within the violation set, then
            # update the color
            if i in violate:
                color = RED

            # draw (1) a bounding box around the person and (2) the
            # centroid coordinates of the person,
            cv2.rectangle(frame, bbox, color, 2)
            cv2.putText(frame, str(od.classes[class_ids[i]]), (startX, startY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, RED)
            cv2.circle(frame, (cX, cY), 5, color, 1)
            print([class_ids[i], bbox])
            bounding_box.append([class_ids[i], bbox])

            video_save_path = "%s\\%s" % (DETECTED_ROOT, 'data.avi')
        if writer is None:  # args["output"] != "" and
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(video_save_path, fourcc, 15,
                                     (frame.shape[1], frame.shape[0]), True)

        # if the video writer is not None, write the frame to the output
        # video file
        if writer is not None:
            writer.write(frame)

    return list(bounding_box)

# Scale width, height of img equal to video
def scale_img_as_video(image, video):
    image = imutils.resize(image,
                           width=int(video.get(3)),
                           height=int(video.get(4)))
    return image

# detect distance in pixel from exacly two people
def detect_distance(image, od):
    inputImage = od.format_yolov5(image)
    (w, h, _) = inputImage.shape
    outs = od.detect(inputImage)
    _, _, boxes = od.wrap_detection(inputImage, outs[0], is_person=True)

    cent = np.array([(int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in boxes])

    dist = sqrt((cent[0][0] - cent[1][0]) ** 2 + (cent[0][1] - cent[1][1]) ** 2)
    return dist