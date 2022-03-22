import pathlib
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework import status, renderers
from library.Detector.object_detector import detect_from_img
from library.Detector.object_tracking import detect_from_video
from mafiaDetector.settings import *
import os

class DetectorView(APIView):
    renderer_classes = [renderers.JSONRenderer]
    """
        Detect friom image, video
    """
    def post(self, request):
        try:
            # Create folder to save img
            detect_path = DETECT_ROOT
            detected_path = DETECTED_ROOT
            # Ensure that the path is created properly and will raise exception if the directory already exists
            if not os.path.exists(detect_path):
                pathlib.Path(detect_path).mkdir(parents=True, exist_ok=True)
                pathlib.Path(detected_path).mkdir(parents=True, exist_ok=True)

            # Initialize variables
            img = ''
            save_video_path = ''
            obj_distance = ''

            ###
            # Check that detect video or detect image
            ###
            # Detect from video
            if request.data.get('video') is not None:
                try:
                    video = request.FILES['video']
                    img_per_real = request.data['img_per_real']
                    # Save video to 'detect' folder
                    save_video_path = "%s\\%s" % (detect_path, video.name)
                    with open(save_video_path, "wb+") as vd:
                        for chunk in video.chunks():
                            vd.write(chunk)

                    # img = request.FILES['img']
                    # Save image to 'detect' folder
                    # save_img_path = "%s\\%s" % (detect_path, img.name)
                    # with open(save_img_path, "wb+") as i:
                    #     for chunk in img.chunks():
                    #         i.write(chunk)
                    detect_option = True
                except Exception as e:
                    return Response({"status": "Wrong video file, image file or distance format"}, status=status.HTTP_400_BAD_REQUEST)
            # Detect from image
            else :
                try:
                    img = request.FILES['img']
                    # Save image to 'detect' folder
                    save_img_path = "%s\\%s" % (detect_path, img.name)
                    with open(save_img_path, "wb+") as f:
                        for chunk in img.chunks():
                            f.write(chunk)
                    detect_option = False
                except Exception as e:
                    return Response({"status": "Wrong image format"}, status=status.HTTP_400_BAD_REQUEST)

            # Detect person from user input
            if detect_option :
                boxes = detect_from_video(save_video_path, img_per_real)
            else:
                boxes = detect_from_img(img.name)
            return Response({'status': 'success', 'data' : boxes}, status=status.HTTP_200_OK)
        except Exception as e:
            print(e)
            return Response({"status": "error"}, status=status.HTTP_404_NOT_FOUND)


