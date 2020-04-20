
from trackableobject import TrackableObject
from centroidtracker import CentroidTracker
from imutils.video import VideoStream, FPS
from io import BytesIO, StringIO
from utils.conf import Conf
from PIL import Image
import numpy as np
import threading
import requests
import argparse
import imutils
import time
import dlib
import cv2
import PIL
import os


def make_request(img):

    ENDPOINT = 'https://eastus.api.cognitive.microsoft.com/face/v1.0/detect'
    KEY = '2d0523e810c24bd5b7fd4448fbf71c67'
    #KEY = '5d4e91c5581544229d0cdc2bc73d89e1'

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

def getRectangle(faceDictionary):
    rect = faceDictionary['faceRectangle']
    left = rect['left']
    top = rect['top']
    right = left + rect['width']
    bottom = top + rect['height']
    
    return ((left, top), (right, bottom))


ct = CentroidTracker()
trackers = []
trackable_objects = {}

W = None
H = None

print("[INFO] starting video stream")
vs = VideoStream(src=0).start()
#time.sleep(2.0)

total_frames = 0
skip_frames = 30
fps = FPS().start()

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=600)   
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    #response = make_request(frame)
    status = "Waiting..."
    rectangles = []

    # Check if object detection has to be performed
    if total_frames % skip_frames == 0:

        response = make_request(frame)
        print("frames passed")
        status = "Detecting..."
        trackers = []
        
        for face in response.json():

            (start_x, start_y), (end_x, end_y) = getRectangle(face)
            box = start_x,start_y,end_x,end_y
            
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(start_x, start_y, end_y, end_y)
            tracker.start_track(rgb, rect)

            trackers.append(tracker)
            #cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,255,0), 2)
            print("detect rect")

    else:
        for tracker in trackers:
            status = "Tracking..."

            tracker.update(rgb)
            pos = tracker.get_position()

            start_x = int(pos.left())
            start_y = int(pos.bottom())
            end_x = int(pos.right())
            end_y = int(pos.top())

            rectangles.append((start_x, start_y, end_x, end_y))
            #cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,255,0), 2)
            print("track rect")

    objects = ct.update(rectangles)
    #print("OBJECT LIST: ", objects)


    for (object_ID, centroid) in objects.items():

        to = trackable_objects.get(object_ID, None)
        if to is None:
            to = TrackableObject(object_ID, centroid)

        trackable_objects[object_ID] = to
        # Draw ID and centroid of the object in the output frame
        text = "ID {}".format(object_ID)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,255,0), 2)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)
        #cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,255,0), 2)

    info = [("Status: ", status)]
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
vs.stop()

