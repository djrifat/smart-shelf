
from utils.detector_utils import get_rectangle
from utils.detector_utils import detect_faces
from utils.detector_utils import make_request
from centroidtracker import CentroidTracker
from imutils.video import VideoStream, FPS
from io import BytesIO, StringIO
from utils.conf import Conf
from PIL import Image
import matplotlib
import numpy as np
import threading
import requests
import argparse
import imutils
import time
import cv2
import PIL
import os

ct = CentroidTracker()
print("[INFO] starting video stream")
cap = cv2.VideoCapture(0)
time.sleep(2.0)

total_frames = 0
skip_frames = 40
fps = FPS().start()

while True:

    ret, frame = cap.read()
    frame = imutils.resize(frame, width=400)    
    response = make_request(frame)
    rectangles = []
    
    for face in response.json():

        face_attribute = face['faceAttributes']
        face_rect = face['faceRectangle']
        face_display = {
            'gender': face_attribute['gender'],
            'age': face_attribute['age']
        }
        face_display.update(face_attribute['emotion'])
        left, top, width, height = face_rect['left'], face_rect['top'], face_rect['width'], face_rect['height']

        (start_x, start_y), (end_x, end_y) = get_rectangle(face)
        box = start_x,start_y,end_x,end_y
        rectangles.append(box)
    
    objects = ct.update(rectangles)

    for (object_ID, centroid) in objects.items():
        # Draw ID and centroid of the object in the output frame
        text = "ID {}".format(object_ID)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,255,0), 1)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        #cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)
        
        for i, k  in enumerate(face_display):
            cv2.putText(frame, "{0}: {1}".format(k, face_display[k]), 
                (left+width+5, top + 5 + 20*i),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    total_frames += 1
    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
ct.total_detections()


cv2.destroyAllWindows()
cap.release()

