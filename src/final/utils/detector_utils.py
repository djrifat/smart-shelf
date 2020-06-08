from azure.cognitiveservices.vision.face.models import APIErrorException
from ratelimit import limits, sleep_and_retry
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
import json
import PIL
import cv2

# Grab frame dimensions
def grab_frame_dim(frame):
    (H, W) = (None, None)
    if H is None or W is None:
        (H, W) = frame.shape[:2]
    return (H,W)

@sleep_and_retry
@limits(calls=5,period=10)
# Make request to MS Face APi
def make_request(buffer_frame):

    ENDPOINT = 'https://eastus.api.cognitive.microsoft.com/face/v1.0/detect'
    KEY = '2d0523e810c24bd5b7fd4448fbf71c67'   
    #KEY = '5d4e91c5581544229d0cdc2bc73d89e1'   # MIRABEAU

    params = {
        'returnFaceId': 'false',
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

    try:
        response = requests.post(data=data,url=ENDPOINT,headers=headers,params=params)
    except Exception as e:
        print('Error: ',  e)

    parsed_response = response.json()
    print(response)
    print(parsed_response)

    if response.status_code == 200:
        parsed_response = response.json()
    else:
        parsed_response = []
    
    return parsed_response

def get_rectangle(face_dict):
    rect = face_dict['faceRectangle']
    left = rect['left']
    top = rect['top']
    right = left + rect['width']
    bottom = top + rect['height']
    
    return ((left, top), (right, bottom))

def unpack_tracker(frame, tracker, rgb, rectangles):
    tracker.update(rgb)
    pos = tracker.get_position()

    start_x = int(pos.left())
    start_y = int(pos.top())
    end_x = int(pos.right())
    end_y = int(pos.bottom())
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,255,0), 1)
    rectangles.append((start_x, start_y, end_x, end_y))
    
    return rectangles

def unpack_rect(frame, tracker, rgb, rectangles):
    tracker.update(rgb)
    pos = tracker.get_position()

    start_x = int(pos.left())
    start_y = int(pos.top())
    end_x = int(pos.right())
    end_y = int(pos.bottom())
    
    return (start_x, start_y, end_x, end_y)


#def combine_captured_data(tracking_ordered_dict, face_dict, combined_dict):


#def save_data_to_csv(data_dict, output_file):
