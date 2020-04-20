
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


def make_request(img):

    ENDPOINT = 'https://eastus.api.cognitive.microsoft.com/face/v1.0/detect'
    KEY = '2d0523e810c24bd5b7fd4448fbf71c67'   
    #KEY = '5d4e91c5581544229d0cdc2bc73d89e1'   # MIRABEAU

    params = {
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': 'age,gender,emotion'
    }

    headers = {
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': KEY
    }

    f = BytesIO()
    PIL.Image.fromarray(img).save(f, 'png')
    data = f.getvalue()

    response = requests.post(data=data,url=ENDPOINT,headers=headers,params=params)
    print(response)
    print(response.json())
    
    return response

def get_rectangle(face_dict):
    rect = face_dict['faceRectangle']
    left = rect['left']
    top = rect['top']
    right = left + rect['width']
    bottom = top + rect['height']
    
    return ((left, top), (right, bottom))

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

        #cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,255,0), 2)
    
    objects = ct.update(rectangles)
    print("OBJECT LIST: ", objects)

    for (object_ID, centroid) in objects.items():
        # Draw ID and centroid of the object in the output frame
        text = "ID {}".format(object_ID)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,255,0), 2)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)
        
        for i, k  in enumerate(face_display):
            cv2.putText(frame, "{0}: {1}".format(k, face_display[k]), (left+width+5, top + 5 + 20*i),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 1)
        
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    total_frames += 1
    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


cv2.destroyAllWindows()
cap.release()

