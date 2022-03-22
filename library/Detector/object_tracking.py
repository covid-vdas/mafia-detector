import cv2
import imutils
import numpy as np
from library.Detector.object_detection import ObjectDetection
from scipy.spatial import distance as dist
from library.Detector.object_config import *
from math import sqrt, hypot
from mafiaDetector.settings import *


def detect_from_video(video_path, img_per_real):

    # Initialize Object Detection
    od = ObjectDetection()

    # Initialize reference video
    ref_video = cv2.VideoCapture(video_path)

    # Get video dimension and set it to image
    ref_video_shape = (int(ref_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(ref_video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Get distance of two person in pixel
    # Besides, we also know real distance of two person in cm(param: obj_distance)
    # So, we can calculate distance of any two person in frame
    # ref_img_width = detect_distance(img_path, ref_video_shape, od)

    # Initialize count
    count = 0

    # Initialize center point of previous frame
    center_points_prev_frame = []

    # Initialize dictionary to save object id : center point of bounding box
    tracking_objects = {}

    # Initialize id for each person
    track_id = 0

    # Create writer to save detected video to folder
    writer = None

    # Save all social distance violations
    # This array save object id : bounding box
    bounding_box = []

    # Loop over video to detect violations
    while True:

        ret, frame = ref_video.read()
        count += 1

        if not ret:
            break

        # Point current frame
        center_points_cur_frame = []

        # Detect all person in each frame
        inputImage = od.format_yolov5(frame)
        outs = od.detect(inputImage)

        # Get all person that confidence over the min constant value
        class_ids, confidences, boxes = od.wrap_detection(inputImage, outs[0], is_person=False)

        # Only get person to tracking object
        for (i, box) in enumerate(boxes):
            if class_ids[i] == 0: 	#Person
                (x, y, w, h) = box
                cx = int((x + x + w) / 2)
                cy = int((y + y + h) / 2)
                center_points_cur_frame.append((cx, cy))
            else:
                if class_ids[i] == 1: # No mask
                    color = BLUE
                    name = 'no'
                    cv2.rectangle(frame, box, BLUE, 2)
                elif class_ids[i] == 2: # Mask
                    color = GREEN
                    name = 'mask'
                    cv2.rectangle(frame, box, GREEN, 2)
                elif class_ids[i] == 3: # Wrong mask
                    color = YELLOW
                    name = 'wrong'
                    cv2.rectangle(frame, box, YELLOW, 2)

                cv2.putText(frame, name, (x + w - 20, y - 20), 0, 0.5, color, 2)

        # Only at the beginning we compare previous and current frame
        if count <= 2:
            for pt in center_points_cur_frame:
                for pt2 in center_points_prev_frame:
                    distance = hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                    # Check that center point in first frame move to new position which is less than 20 pixels
                    if distance < 20:
                        tracking_objects[track_id] = pt
                        track_id += 1
        else:
            tracking_objects_copy = tracking_objects.copy()
            center_points_cur_frame_copy = center_points_cur_frame.copy()

            for object_id, pt2 in tracking_objects_copy.items():
                object_exists = False
                for pt in center_points_cur_frame_copy:
                    distance = hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                    # Check that center point in first frame move to new position which is less than 20 pixels
                    if distance < 30:
                        tracking_objects[object_id] = pt
                        object_exists = True
                        if pt in center_points_cur_frame:
                            center_points_cur_frame.remove(pt)
                        continue

                # Remove IDs lost
                if not object_exists:
                    tracking_objects.pop(object_id)

            # Add new IDs found
            for pt in center_points_cur_frame:
                tracking_objects[track_id] = pt
                track_id += 1

        # Get all center points in frame
        # To detect social violations
        center_points = np.array([(int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in boxes])
        if len(center_points) != 0:

            D = dist.cdist(center_points, center_points, metric="euclidean")
            # Set to save all violations that can be duplicated
            violate = set()

            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # Check to see if the distance between any two
                    # centroid pairs is less than the configured number
                    # of centimeter

                    # 138 pixel -> 500 cm
                    # D[i, j] pixel -> 200 cm
                    # distance = D[i, j] * KNOW_WIDTH / ref_img_width
                    if D[i, j] < MIN_DISTANCE * int(img_per_real):
                        (x1, y1, w1, h1) = boxes[i]
                        (x2, y2, w2, h2) = boxes[j]
                        cent1 = [int(x1 + w1 / 2), int(y1 + h1 / 2)]
                        cent2 = [int(x2 + w2 / 2), int(y2 + h2 / 2)]

                        distance = D[i, j] * int(img_per_real) / 100
                        distance = round(distance, 2)

                        cv2.line(frame, cent1, cent2, RED, 2)
                        cv2.putText(frame, str(distance), (cent1[0], cent1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, RED)
                        violate.add(i)
                        violate.add(j)
        # Loop all bounding box to get object id and center point
        # to draw necessary information and save violations
        for (i, (bbox)) in enumerate(boxes):
            # extract the bounding box and centroid coordinates, then
            # initialize the color of the annotation
            ob_id = -1
            color = GREEN
            (startX, startY, width, height) = bbox
            cent = (int((startX + startX + width) / 2), int((startY + startY + height) / 2))

            # if the index pair exists within the violation set, then
            # update the color
            for object_id, pt in tracking_objects.items():
                if(pt[0] == cent[0] and pt[1] == cent[1]):
                    ob_id = object_id
                    break
            if i in violate:
                bounding_box.append([ob_id, bbox])
                color = RED

            # draw (1) a bounding box around the person and (2) the
            # centroid coordinates of the person,
            cv2.putText(frame, str(ob_id), (cent[0], cent[1]), 0, 0.5, color, 2)
            cv2.rectangle(frame, bbox, color, 2)
            cv2.putText(frame, str(confidences[i]), (startX, startY - 20), 0, 0.5, color, 2)

        # Save detected video
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

        # Make a copy of the points
        center_points_prev_frame = center_points_cur_frame.copy()

    ref_video.release()
    return bounding_box

def detect_from_stream(frame, od):

    # Detect all person in each frame
    inputImage = od.format_yolov5(frame)
    outs = od.detect(inputImage)

    # Get all person that confidence over the min constant value
    class_ids, confidences, boxes = od.wrap_detection(inputImage, outs[0], is_person=False)

    # Loop all bounding box to get object id and center point
    # to draw necessary information and save violations
    for (i, (bbox)) in enumerate(boxes):

        color = GREEN
        (startX, startY, width, height) = bbox

        cv2.rectangle(frame, bbox, color, 2)
        cv2.putText(frame, str(od.classes[class_ids[i]]), (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, .5, GREEN)
        cv2.putText(frame, str(confidences[i]), (startX + int(width/2), startY - 20), 0, 0.5, color, 2)

    return frame

# Scale width, height of img equal to video
def scale_img_as_video(image, video):
    image = imutils.resize(image,
                           width=int(video.get(3)),
                           height=int(video.get(4)))
    return image

# detect distance in pixel from exacly two people
def detect_distance(img_name, video, od):
    # Ensure that in img have exactly two person
    image = cv2.imread(str(DETECT_ROOT / img_name))
    # Scale width, height of img suitable with video
    image = cv2.resize(image, video)

    # Initialize reference image from user input
    ref_image = od.format_yolov5(image)
    outs = od.detect(ref_image)
    _, _, boxes = od.wrap_detection(ref_image, outs[0], is_person=True)

    # Calculate distance between two person
    cent = np.array([(int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in boxes])
    dist = sqrt((cent[0][0] - cent[1][0]) ** 2 + (cent[0][1] - cent[1][1]) ** 2)

    # Draw necessary information in image
    for box in boxes:
        cv2.rectangle(image, box, GREEN, 2)
        cv2.line(image, cent[0], cent[1], GREEN, 1)
        cv2.putText(image, str(round(dist, 2)),
                    (int((cent[0][0] + cent[1][0]) / 2), int((cent[0][1] + cent[1][1]) / 2 - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, RED, 2)

    # Save detected img to detected folder
    cv2.imwrite(str(DETECTED_ROOT / img_name), image)

    return dist