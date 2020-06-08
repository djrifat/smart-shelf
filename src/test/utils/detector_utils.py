
from imutils.video import VideoStream, FPS
from io import BytesIO, StringIO
from threading import Thread
from PIL import Image
import numpy as np
import argparse
import requests
import imutils
import sched
import time
import PIL
import cv2

# Grab frame dimensions
def grab_frame_dim(frame):
    (H, W) = (None, None)
    if H is None or W is None:
        (H, W) = frame.shape[:2]
    return (H,W)

# Make request to MS Face APi
def make_request(buffer_frame):

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
    PIL.Image.fromarray(buffer_frame).save(f, 'png')  
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

def unpack_tracker(frame, tracker, rgb, rects):
    tracker.update(rgb)
    pos = tracker.get_position()

    start_x = int(pos.left())
    start_y = int(pos.top())
    end_x = int(pos.right())
    end_y = int(pos.bottom())
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,0,255), 1)
    rects.append((start_x, start_y, end_x, end_y))
    
    return rects



