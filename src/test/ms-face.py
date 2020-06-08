
from centroidtracker import CentroidTracker
from imutils.video import VideoStream, FPS
from io import BytesIO, StringIO
from utils.conf import Conf
import utils.detector_utils
from PIL import Image
import matplotlib
import numpy as np
import threading
import requests
import argparse
import operator
import imutils
import dlib
import time
import cv2
import PIL
import os

ct = CentroidTracker()
total_frames = 0
skip_frames = 60
total_frames = 0
trackers = []

print("[INFO] starting video stream")
cap = cv2.VideoCapture(0)
time.sleep(2.0)
fps = FPS().start()

while True:

    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    frame = imutils.resize(frame, width=500)   
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    (H,W) = utils.detector_utils.grab_frame_dim(frame)
    
    rectangles = []
    status = "Waiting..."
    
    if total_frames % skip_frames == 0:

        status = "Detecting..."
        trackers = []
        response = utils.detector_utils.make_request(frame)

        for face in response.json():

            face_attribute = face['faceAttributes']
            face_rect = face['faceRectangle']

            left, top, width, height = face_rect['left'], face_rect['top'], face_rect['width'], face_rect['height']
            (start_x, start_y), (end_x, end_y) = utils.detector_utils.get_rectangle(face)

            t = dlib.correlation_tracker()
            rect = dlib.rectangle(start_x,start_y,end_x,end_y)
            t.start_track(rgb, rect)
            trackers.append(t)  
        
    else:
        for t in trackers:
            status = "Tracking..."
            utils.detector_utils.unpack_tracker(frame, t, rgb, rectangles)

    objects = ct.update(rectangles)

    for (object_ID, centroid) in objects.items():
        # Draw ID and centroid of the object in the output frame
        text = "ID {}".format(object_ID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        for face in response.json():

            face_attribute = face['faceAttributes']
            face_rect = face['faceRectangle']
            emotions = face['faceAttributes']['emotion']
            current_mood = max(emotions.items(), key=operator.itemgetter(1))[0]
            left, top, width, height = face_rect['left'], face_rect['top'], face_rect['width'], face_rect['height']

            face_display = {
                'gender': face_attribute['gender'],
                'age': face_attribute['age'],
                'mood': current_mood
            }

            #for i, k  in enumerate(face_display):
            for k, v in face_display.items():
                #print(k, face_display[k])
                print (k, v)
                #print("ZERO: ",centroid[0])
                #print("ONE: ", k, face_display[k])
                #text2 = "{0}: {1}".format(v, face_display[])
                cv2.putText(frame, "{0}: {1}".format(v, face_display[v]), 
                    (centroid[0]+5, centroid[1] + 5 + 20*k),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA) 
    
    info = [("Status: ", status)]
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

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

#(start_x, start_y), (end_x, end_y) = [[left, top], [left + width, top], [left + width, top + height], [left, top + height]]
