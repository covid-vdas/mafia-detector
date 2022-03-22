from library.Detector.object_detection import ObjectDetection
from library.Detector.object_tracking import detect_from_stream
from django.http.response import StreamingHttpResponse
from rest_framework.views import APIView
from rest_framework import status, renderers
from rest_framework.response import Response
import cv2

from mafiaDetector.settings import DETECTED_ROOT


class StreamView(APIView):
    renderer_classes = [renderers.JSONRenderer]

    def stream_from_url(self, stream_url):

        ref_video = cv2.VideoCapture(stream_url)
        writer = None
        count = 0
        od = ObjectDetection()
        while True:
            ret, frame = ref_video.read()

            count += 1
            if ret:
                frame = detect_from_stream(frame, od)
                print(f"Creating file... {count}")
                # cv2.imwrite(str(DETECTED_ROOT) + '\\frame.jpg', frame)

            if writer is None:  # args["output"] != "" and
                # initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(str(DETECTED_ROOT) + '\\frame.avi', fourcc, 10,
                                         (frame.shape[1], frame.shape[0]), True)

            # if the video writer is not None, write the frame to the output
            # video file
            if writer is not None:
                writer.write(frame)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open(str(DETECTED_ROOT) + '\\frame.jpg', 'rb').read() + b'\r\n')

    def post(self, request):
        try:
            stream_url = request.data['url']
            print(stream_url)
            return StreamingHttpResponse(self.stream_from_url(str(stream_url)),
                                         content_type='multipart/x-mixed-replace; boundary=frame', status=200)
        except  Exception as e:
            return StreamingHttpResponse(e, content_type='text', status=404)
