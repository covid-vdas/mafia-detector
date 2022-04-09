import sys
sys.path.insert(0, './yolov5')

import torch
import torch.backends.cudnn as cudnn
from mafiaDetector.settings import *
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
								  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from config import *

def detect(path):
	source = path
	device = select_device()
	cfg = get_config()
	cfg.merge_from_file(DEEP_SORT_YAML_PATH)

	deepsort = DeepSort(DEEP_SORT_MODEL,
						device,
						max_dist=cfg.DEEPSORT.MAX_DIST,
						max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
						max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
						)
	half = True
	# Initialize
	half &= device.type != 'cpu'  # half precision only supported on CUDA

	Path(DETECTED_ROOT).mkdir(parents=True, exist_ok=True)# make new output folder
	
	# Directories
	if type(YOLO_MODEL) is str:
		exp_name = YOLO_MODEL.split(".")[0]
	elif type(YOLO_MODEL) is list and len(YOLO_MODEL) == 1:
		exp_name = YOLO_MODEL[0].split(".")[0]
	else:
		exp_name = "ensemble"
	exp_name = exp_name + "_" + DEEP_SORT_MODEL.split('/')[-1].split('.')[0]
	save_dir = increment_path(Path(DETECTED_ROOT) / exp_name, exist_ok=True)  # increment run if project name exists
	save_dir.mkdir(parents=True, exist_ok=True)  # make dir

	# Load model
	device = select_device()
	model = DetectMultiBackend(YOLO_MODEL, device=device, dnn=True)
	stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
	imgsz = check_img_size([INPUT_WIDTH, INPUT_HEIGHT], s=stride)  # check image size

	# Half
	half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
	if pt:
		model.model.half() if half else model.model.float()

	# show_vid = check_imshow()

	# Dataloader 
	cudnn.benchmark = True
	dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
	# ref_video = cv2.VideoCapture(path)  

	bs = len(dataset)
	vid_path, vid_writer = [None] * bs, [None] * bs

	# Get names and colors
	names = model.module.names if hasattr(model, 'module') else model.names
  
	if pt and device.type != 'cpu':
		model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
	dt, seen = [0.0, 0.0, 0.0, 0.0], 0

	for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
		t1 = time_sync()
		img = torch.from_numpy(img).to(device)
		img = img.half() if half else img.float()  # uint8 to fp16/32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)
		t2 = time_sync()
		dt[0] += t2 - t1

		# prediction after detection
		pred = model(img, augment=True, visualize=False)
		t3 = time_sync()
		dt[1] += t3 - t2

		# Apply NMS 
		# 
		# conf_thres: object confidence threshold
		# iou_thres	: IOU threshold for NMS
		# classes	: filter by class: --class 0, or --class 16 17
		# agnostic	: class-agnostic NMS
		# max_det	: maximum detection per image
		pred = non_max_suppression(prediction=pred, conf_thres=0.5, iou_thres=0.5, classes=None, agnostic=False, max_det=1000)
		dt[2] += time_sync() - t3

		# Process detections
		# detections per image
		for i, det in enumerate(pred):

			im0 = im0s[i].copy()
			annotator = Annotator(im0, line_width=2, pil=not ascii)

			if det is not None and len(det):
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(
					img.shape[2:], det[:, :4], im0.shape).round()

				xywhs = xyxy2xywh(det[:, 0:4])
				confs = det[:, 4]
				clss = det[:, 5]

				# pass detections to deepsort
				t4 = time_sync()

				#pass detection bounding box to deep sort to tracking
				outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
				t5 = time_sync()
				dt[3] += t5 - t4

				# draw boxes for visualization
				if len(outputs) > 0:
					for j, (output, conf) in enumerate(zip(outputs, confs)):

						bboxes = output[0:4]
						id = output[4]
						cls = output[5]

						c = int(cls)  # integer class
						label = f'{id} {names[c]} {conf:.2f}'
						annotator.box_label(bboxes, label, color=colors(c, True))

				LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

			else:
				deepsort.increment_ages()
				LOGGER.info('No detections')

			# Stream results
			im0 = annotator.result()










