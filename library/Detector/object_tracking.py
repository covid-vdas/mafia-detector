import cv2
import numpy as np
from object_detection import ObjectDetection
from scipy.spatial import distance as dist
from object_config import *
import math

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture(DATA_PATH)

# Initialize count
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

while True:
	ret, frame = cap.read()
	count += 1
	if not ret:
		break

	# Point current frame
	center_points_cur_frame = []

	inputImage = od.format_yolov5(frame)
	outs = od.detect(inputImage) 

	# Detect person in frame
	class_ids, confidences, boxes = od.wrap_detection(inputImage, outs[0], is_person = True)
	
	# label all person 
	for (i, box) in enumerate(boxes):
	    if(class_ids[i] == 0):
	        (x, y, w, h) = box
	        cx = int((x + x + w) / 2)
	        cy = int((y + y + h) / 2)
	        center_points_cur_frame.append((cx, cy)) 

	# Only at the beginning we compare previous and current frame
	if count <= 2:
		for pt in center_points_cur_frame:
			for pt2 in center_points_prev_frame:
				distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

				if distance < 20:
					tracking_objects[track_id] = pt
					track_id += 1
	else: 
		tracking_objects_copy = tracking_objects.copy()
		center_points_cur_frame_copy = center_points_cur_frame.copy() 

		for object_id, pt2 in tracking_objects_copy.items():
			object_exists = False
			for pt in center_points_cur_frame_copy:
				distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1]) 
				# Update IDs position
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

	center_points = np.array([(int(x + w/2), int(y + h/2)) for (x, y, w, h) in boxes])
	if len(center_points) != 0:

		D = dist.cdist(center_points, center_points, metric="euclidean")

		violate = set()
 
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two
				# centroid pairs is less than the configured number
				# of centimeter

				# 138 pixel -> 500 cm
				# D[i, j] pixel -> 200 cm
				# distance = D[i, j] * KNOW_WIDTH / ref_img_width 
				if D[i, j] < MIN_DISTANCE * 405 / OBJECT_DISTANCE:
					# update our violation set with the indexes of
					# the centroid pairs

					(x1, y1, w1, h1) = boxes[i]
					(x2, y2, w2, h2) = boxes[j]
					cent1 = [int(x1 + w1/2), int(y1 + h1/2)]
					cent2 = [int(x2 + w2/2), int(y2 + h2/2)]
					# print(cent1, ", ", cent2)

					cv2.line(frame, cent1, cent2, RED, 2)
					distance = D[i, j] * OBJECT_DISTANCE / 405 / 100

					distance = round(distance, 2)

					# cv2.putText(frame, str(distance), (cent1[0], cent1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, RED)
					violate.add(i)
					violate.add(j)
	 
	for (i, (bbox)) in enumerate(boxes):
		# extract the bounding box and centroid coordinates, then
		# initialize the color of the annotation
		(startX, startY, width, height) = bbox
		(cX, cY) = center_points[i]
		color = GREEN

		# if the index pair exists within the violation set, then
		# update the color
		if i in violate:
			color = RED

		# draw (1) a bounding box around the person and (2) the
		# centroid coordinates of the person,
		cv2.rectangle(frame, bbox, color, 2)
		# cv2.circle(frame, (cX, cY), 5, color, 1)
 

	# print("tracking_objects: ", tracking_objects)
	for object_id, pt in tracking_objects.items():
		# cv2.circle(frame, pt, 5, (0, 0, 255), -1)
		cv2.putText(frame, str(object_id), (pt[0], pt[1] - 30), 0, 1, (0, 0, 255), 1)
	 
	cv2.imshow("Frame", frame)

	# Make a copy of the points
	center_points_prev_frame = center_points_cur_frame.copy()

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()
