from django.http.response import StreamingHttpResponse
from rest_framework.views import APIView
from rest_framework import status, renderers
from rest_framework.response import Response
import cv2
import numpy as np
from scipy.spatial import distance as dist
from datetime import datetime
from PIL import Image as pil_img
import torch
import torch.backends.cudnn as cudnn
from mafiaDetector.settings import *
from library.Detector.Yolov5_DeepSort_Pytorch.yolov5.models.common import DetectMultiBackend
from library.Detector.Yolov5_DeepSort_Pytorch.yolov5.utils.datasets import LoadImages, LoadStreams
from library.Detector.Yolov5_DeepSort_Pytorch.yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
								  check_imshow, xyxy2xywh, increment_path)
from library.Detector.Yolov5_DeepSort_Pytorch.yolov5.utils.torch_utils import select_device, time_sync
from library.Detector.Yolov5_DeepSort_Pytorch.yolov5.utils.plots import Annotator, colors
from library.Detector.Yolov5_DeepSort_Pytorch.deep_sort.utils.parser import get_config
from library.Detector.Yolov5_DeepSort_Pytorch.deep_sort.deep_sort import DeepSort
from library.Detector.Yolov5_DeepSort_Pytorch.config import *

from api.models.violation_model import Violation
from api.models.violation_type_model import ViolationType
from api.models.camera_model import Camera
from api.models.image_model import Image
from api.models.object_information_model import ObjectInformation

from api.serializer import *



class DetectorView(APIView):
    renderer_classes = [renderers.JSONRenderer]

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

    # Load model
    device = select_device()
    model = DetectMultiBackend(YOLO_MODEL, device=device, dnn=True)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size([INPUT_WIDTH, INPUT_HEIGHT], s=stride)  # check image size

    # make new output folder
    Path(DETECT_ROOT).mkdir(parents=True, exist_ok=True)
    Path(DETECTED_ROOT).mkdir(parents=True, exist_ok=True)
    Path(str(DETECTED_ROOT) + '\\yeild').mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    # Initialize
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    def detect(self, path, ratio):
        try:
            source = path
            # Dataloader
            dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
            bs = 1  # batch_size
            vid_path, vid_writer = [None] * bs, [None] * bs

            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            # Run inference
            # self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.imgsz), half=self.half)  # warmup
            dt, seen = [0.0, 0.0, 0.0], 0
            for path, im, im0s, vid_cap, s in dataset:
                t1 = time_sync()
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.half else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                t2 = time_sync()
                dt[0] += t2 - t1

                # Inference
                pred = self.model(im, augment=True, visualize=False)
                t3 = time_sync()
                dt[1] += t3 - t2

                # NMS
                pred = non_max_suppression(prediction=pred, conf_thres=0.5, iou_thres=0.5, classes=None, agnostic=False,
                                           max_det=1000)
                dt[2] += time_sync() - t3

                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(DETECTED_ROOT / p.name)  # im.jpg
                    annotator = Annotator(im0, line_width=2, example=str(names), pil=not ascii)
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                        # Write results
                        boxes = det[:, 0:4]
                        confs = det[:, 4]
                        clss = det[:, 5]
                        center_points = np.array([(int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in boxes])
                        if len(center_points) != 0:

                            D = dist.cdist(center_points, center_points, metric="euclidean")
                            # Set to save all violations that can be duplicated
                            violate = set()

                            for k in range(0, D.shape[0]):
                                for j in range(k + 1, D.shape[1]):
                                    # Check to see if the distance between any two
                                    # centroid pairs is less than the configured number
                                    # of centimeter

                                    # 138 pixel -> 500 cm
                                    # D[i, j] pixel -> 200 cm
                                    # distance = D[i, j] * KNOW_WIDTH / ref_img_width
                                    if D[k, j] < MIN_DISTANCE / float(ratio):
                                        # (x1, y1, w1, h1) = boxes[k]
                                        # (x2, y2, w2, h2) = boxes[j]
                                        # cent1 = [int(x1 + w1 / 2), int(y1 + h1 / 2)]
                                        # cent2 = [int(x2 + w2 / 2), int(y2 + h2 / 2)]
                                        #
                                        # distance = D[k, j] * float(ratio) / 100
                                        # distance = round(distance, 2)
                                        # annotator.draw.line((cent1, cent2))
                                        # annotator.text((cent1[0], cent1[1] - 5), str(distance), cv2.FONT_HERSHEY_SIMPLEX)
                                        # cv2.line(im0, cent1, cent2, RED, 2)
                                        # cv2.putText(im0, str(distance), (cent1[0], cent1[1] - 5),
                                        #             cv2.FONT_HERSHEY_SIMPLEX, .5, RED)
                                        violate.add(k)
                                        violate.add(j)

                        # Loop all bounding box to get object id and center point
                        # to draw necessary information and save violations
                        for l, (bbox) in enumerate(boxes):
                            # extract the bounding box and centroid coordinates, then
                            # initialize the color of the annotation
                            color = GREEN
                            (startX, startY, width, height) = bbox
                            # if the index pair exists within the violation set, then
                            # update the color
                            if l in violate:
                                color = RED
                            if names[int(clss[l])] != 'person' and names[int(clss[l])] != 'mask':
                                color = RED
                            # draw (1) a bounding box around the person and (2) the
                            # centroid coordinates of the person,
                            annotator.box_label(bbox, str(names[int(clss[l])] + ' ' + str(round(float(confs[l]), 2))), color)
                        #
                        # c = int(cls)  # integer class
                        # label = f'{names[c]} {conf:.2f}'
                        # annotator.box_label(xyxy, label, color=GREEN)

                    # Stream results
                    im0 = annotator.result()

                    # Save results (image with detections)
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.avi'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)
        except Exception as e:
            return False
        return True

    def detect_from_stream(self, path, ratio, type_obj = None, camera_id = ''):
        violate_dict = dict()
        source = path

        # Dataloader
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride, auto=self.pt and not self.jit)
        bs = len(dataset)  # batch_size

        # Get names
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        if self.pt and self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.imgsz).to(self.device).type_as(next(self.model.model.parameters())))  # warmup
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0

        for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
            t1 = time_sync()
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            # visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if opt.visualize else False
            pred = self.model(img, augment=True, visualize=False)
            t3 = time_sync()
            dt[1] += t3 - t2

            # Apply NMS
            pred = non_max_suppression(prediction=pred, conf_thres=0.5, iou_thres=0.5, classes=None, agnostic=False,
                                       max_det=1000)
            dt[2] += time_sync() - t3

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                seen += 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                annotator = Annotator(im0, line_width=2, pil=not ascii)

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to deepsort
                    t4 = time_sync()
                    outputs = self.deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    t5 = time_sync()
                    dt[3] += t5 - t4

                    # draw boxes for visualization
                    if len(outputs) > 0:

                        violate_dict = self.tracking_violate(outputs, ratio, violate_dict, type_obj)
                        for j, (output, conf) in enumerate(zip(outputs, confs)):

                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            c = int(cls)  # integer class

                            color = GREEN
                            print(violate_dict)
                            if violate_dict is not None and id in violate_dict.keys():
                                if violate_dict[id] >= CONF_VIO_CONTINUOUS_FRAME and cls == 0:
                                    list_person_violate = []
                                    img_person_violate = im0.copy()
                                    annotator_img_person_violate = Annotator(img_person_violate, line_width=2,
                                                                             pil=not ascii)
                                    for out in outputs:
                                        if out[5] == 0:
                                            if self.cal_distance_img(bboxes, out[0:4]) <= MIN_DISTANCE * ratio:
                                                list_person_violate.append([out[4], out[0:4]])
                                                violate_dict[out[4]] = -1

                                    violate_dict[id] = -1
                                    for (id_person_violate, bbox) in list_person_violate:
                                        label_img_person_violate = f'{id_person_violate} person {conf:.2f}'

                                        annotator_img_person_violate.box_label(bbox, label, color=RED)
                                    file_name = str('Distance violation ' + str(datetime.now()) + '.png')
                                    save_img_path = "%s\\%s" % (DETECTED_ROOT, file_name)
                                    label = f'{id} {names[c]} {conf:.2f}'
                                    annotator.box_label(bboxes, label, color=RED)
                                    saved_img = annotator_img_person_violate.result()  # RGB img
                                    saved_img = saved_img[..., ::-1]  # convert RGB to BGR img
                                    img = pil_img.fromarray(saved_img, 'RGB')  # format BRG img to Image
                                    img.save(save_img_path)
                                    distance_img = self.cal_distance_img(list_person_violate[0][1], list_person_violate[1][1])
                                    distance_real = distance_img / ratio
                                    if isinstance(distance_real, np.generic):
                                        distance_real = np.asscalar(distance_real)

                                    Image.objects.create(name = str('Distance violaion ' + str(datetime.now()).replace(':', '-')),
                                                         url = save_img_path)
                                    ##luu db
                                    Violation.objects.create(type_id = ViolationType.objects(name = 'Distance').first().id,
                                                             camera_id = str(Camera.objects(id = camera_id).first().id),
                                                             image_id = Image.objects(url = save_img_path).first().id,
                                                             class_id = ObjectInformation.objects(cardinality = c).first().id,
                                                             distance = str(distance_real))

                                if violate_dict[id] >= CONF_VIO_CONTINUOUS_FRAME and cls != 0:
                                    file_name = str('Distance violation ' + str(datetime.now()).replace(':', '-') + '.png')
                                    save_img_path = "%s\\%s" % (DETECTED_ROOT, file_name)
                                    label = f'{id} {names[c]} {conf:.2f}'
                                    annotator.box_label(bboxes, label, color=RED)
                                    saved_img = annotator.result()  # RGB img
                                    saved_img = saved_img[..., ::-1]  # convert RGB to BGR img
                                    img = pil_img.fromarray(saved_img, 'RGB')  # format BRG img to Image
                                    img.save(save_img_path)

                                    violate_dict[id] = -1

                                    Image.objects.create(name=str('Distance violaion ' + str(datetime.now())),
                                                         url=save_img_path)

                                    #luu db

                                    Violation.objects.create(type_id = ViolationType.objects(name='Facemask').first().id,
                                                             camera_id = str(Camera.objects(id = camera_id).first().id) ,
                                                             image_id = Image.objects(url=save_img_path).first().id,
                                                             class_id = ObjectInformation.objects(cardinality = c).first().id)
                                    continue
                                color = RED

                            if type_obj is None:  # person and mask
                                label = f'{id} {names[c]} {conf:.2f}'
                                annotator.box_label(bboxes, label, color=color)
                            elif type_obj == 'mask':  # only mask detection
                                if cls != 0:
                                    label = f'{id} {names[c]} {conf:.2f}'
                                    annotator.box_label(bboxes, label, color=color)
                            elif type_obj == 'person':  # only person detection
                                if cls == 0:
                                    label = f'{id} {names[c]} {conf:.2f}'
                                    annotator.box_label(bboxes, label, color=color)
                else:
                    self.deepsort.increment_ages()
                    LOGGER.info('No detections')

                # Stream results
                im0 = annotator.result()

                cv2.imwrite(str(DETECTED_ROOT) + '\\yeild\\img.jpg', im0)
                img_path = str(DETECTED_ROOT) + '\\yeild\\img.jpg'
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open(img_path, 'rb').read() + b'\r\n')

    # ['person', 'no mask', 'mask', 'wrong mask']
    def tracking_violate(self, outputs, ratio, violate_dict, type_obj):

        boxes = []

        for output in outputs:
            if type_obj is None:  # person and mask
                boxes.append([output[:4], output[4], output[5]])
            elif type_obj == 'mask':  # only mask detection
                if output[5] != 0:
                    boxes.append([output[:4], output[4], output[5]])
            elif type_obj == 'person':  # only person detection
                if output[5] == 0:
                    boxes.append([output[:4], output[4], output[5]])

        ids = np.array([box[1] for box in boxes])
        cent = []
        # pop cua mask
        temp_violate_dict = violate_dict.copy()
        for vio in temp_violate_dict.keys():
            if vio not in ids:
                violate_dict.pop(vio, None)
            if vio in ids and boxes[list(ids).index(vio)][2] == 2:
                violate_dict.pop(vio, None)

        for i, ((x, y, w, h), id_obj, c) in enumerate(boxes):
            if c == 0:
                cent.append((int(x + w / 2), int(y + h / 2)))
            else:
                if violate_dict is not None and c != 2:
                    if id_obj in violate_dict.keys() and violate_dict[id_obj] != -1:
                        violate_dict[id_obj] = violate_dict[id_obj] + 1
                    elif id_obj in violate_dict.keys() and violate_dict[id_obj] == -1:
                        continue
                    else:
                        violate_dict[id_obj] = 1
        if len(cent) != 0:
            D = dist.cdist(cent, cent, metric="euclidean")
            temp_vio_person = set()
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # check to see if the distance between any two
                    # centroid pairs is less than the configured number
                    # of centimeter

                    # 138 pixel -> 500 cm
                    # D[i, j] pixel -> 200 cm
                    # distance = D[i, j] * KNOW_WIDTH / ref_img_width
                    # print("D[i, j]: ", D[i, j])

                    if D[i, j] < MIN_DISTANCE * ratio:  # ref_img_width / OBJECT_DISTANCE

                        if violate_dict is not None:
                            # distance = D[i, j] / ratio #real distance between two people

                            if boxes[i][1] in violate_dict.keys() and violate_dict[boxes[i][1]] != -1:
                                violate_dict[boxes[i][1]] = violate_dict[boxes[i][1]] + 1
                            elif boxes[i][1] not in violate_dict.keys():
                                violate_dict[boxes[i][1]] = 1

                            if boxes[j][1] in violate_dict.keys() and violate_dict[boxes[j][1]] != -1:
                                violate_dict[boxes[j][1]] = violate_dict[boxes[j][1]] + 1
                            elif boxes[j][1] not in violate_dict.keys():
                                violate_dict[boxes[j][1]] = 1
                            temp_vio_person.add(boxes[i][1])
                            temp_vio_person.add(boxes[j][1])

            for idx, wrong_id_person in enumerate(ids):
                if wrong_id_person not in temp_vio_person and boxes[idx][2] == 'person':
                    violate_dict.pop(wrong_id_person, None)

        print(violate_dict)
        return violate_dict

    def cal_distance_img(self, box1, box2):
        x1 = box1[0] + box1[2] / 2
        y1 = box1[1] + box1[3] / 2

        x2 = box2[0] + box2[2] / 2
        y2 = box2[1] + box2[3] / 2
        return (((x2 - x1) ** 2) + ((y2 - y1) ** 2)) ** 0.5

    def cal_distance_real(self, distance_img, ratio):
        return distance_img / ratio

    def get(self, request):
        try:
            input_type = 0 # 1 img, 2 vid, 3 stream
            save_input_video_path = ''
            save_input_img_path = ''

            ###
            # Check that detect video or detect image
            ###
            # Detect from video
            if request.data.get('video') is not None:
                try:
                    video = request.FILES['video']
                    ratio = request.data['ratio']
                    # Save video to 'detect' folder
                    save_input_video_path = "%s\\%s" % (DETECT_ROOT, video.name)
                    with open(save_input_video_path, "wb+") as vd:
                        for chunk in video.chunks():
                            vd.write(chunk)
                    input_type = 1
                except Exception as e:
                    return Response({"status": "Wrong video file, image file or distance format"},
                                    status=status.HTTP_400_BAD_REQUEST)
            # Detect from image
            elif request.data.get('img') is not None:
                try:
                    img = request.FILES['img']
                    ratio = request.data['ratio']
                    # Save image to 'detect' folder
                    save_input_img_path = "%s\\%s" % (DETECT_ROOT, img.name)
                    with open(save_input_img_path, "wb+") as f:
                        for chunk in img.chunks():
                            f.write(chunk)
                    input_type = 2
                except Exception as e:
                    return Response({"status": "Wrong image format"}, status=status.HTTP_400_BAD_REQUEST)
            elif request.data.get('stream_url') is not None:
                try:
                    stream_url = request.data['stream_url']
                    ratio = request.data['ratio']
                    obj_detect_type = request.data['obj_detect_type']
                    camera_id= request.data['camera_id']
                    input_type = 3
                except Exception as e:
                    return Response({"status": "Wrong input"}, status=status.HTTP_400_BAD_REQUEST)


            if input_type == 1:
                if self.detect(save_input_video_path, ratio):
                    return Response({'status': 'success'}, status=status.HTTP_200_OK)
            elif input_type == 2:
                if self.detect(save_input_img_path, ratio):
                    return Response({'status': 'success'}, status=status.HTTP_200_OK)
            elif input_type == 3:

                return StreamingHttpResponse(self.detect_from_stream(path=stream_url,
                                                         ratio=float(ratio),
                                                         type_obj=str(obj_detect_type),
                                                         camera_id = str(camera_id)),
                                             content_type='multipart/x-mixed-replace; boundary=frame', status=200)

            return Response({'status': 'fail'}, status=status.HTTP_404_NOT_FOUND)
        except  Exception as e:
            return Response({'status': 'fail'}, status=status.HTTP_404_NOT_FOUND)
