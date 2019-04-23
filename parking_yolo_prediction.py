from opencv.base_camera import BaseCamera
import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

class Camera(BaseCamera):
    video_source = 0

    @staticmethod
    def set_video_source(source):
        Camera.video_source = 'videos/testvideo.mp4'

    @staticmethod
    def frames():

        options = {
            'model': 'cfg/yolo.cfg',
            'load': 'bin/yolo.weights',
            'threshold': 0.2,
            'gpu': 0.5
        }

        tfnet = TFNet(options)
        colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

        capture = cv2.VideoCapture('rtsp://192.168.0.23:554/12')
        #capture = cv2.VideoCapture('videos/testvideo.mp4')
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # print(coordinates.shape)

        while True:
            stime = time.time()
            ret, frame = capture.read()

            label_counting = 0
            label_confidence = 0

            if ret:
                results = tfnet.return_predict(frame)
                for color, result in zip(colors, results):
                    tl = (result['topleft']['x'], result['topleft']['y'])
                    br = (result['bottomright']['x'], result['bottomright']['y'])
                    label = result['label']
                    confidence = result['confidence']

                    label_confidence = confidence * 100

                    if label == "person" and label_confidence > 70:
                        label_counting = label_counting + 1

                    text = '{}: {:.0f}%'.format(label, confidence * 100)
                    frame = cv2.rectangle(frame, tl, br, color, 5)
                    frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                print('FPS {:.1f}'.format(1 / (time.time() - stime)))
                print('How many people in this image? {}'.format(label_counting))

            yield cv2.imencode('.jpg', frame)[1].tobytes()
