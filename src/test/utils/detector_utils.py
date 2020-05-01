
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

# Grab frame dimensions
def grab_frame_dim(frame):
    (H, W) = (None, None)
    if H is None or W is None:
        (H, W) = frame.shape[:2]
    return (H,W)

def detect_faces(frame, net, ct):
    # Grab frame dimensions
    # Create a blob from the frame, pass the frame through 
    # the CNN to obtain predections and initialize list of bounding box rectangles
    (H,W) = grab_frame_dim(frame)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W,H), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rectangles = []
    
    # Process detections
    # Loop through detections
    for i in np.arange(0, detections.shape[2]):
        # Filter out weak detections
        # Ensure predicted probability is greater then minimum threshold
        if detections[0, 0, i ,2] > 0.5:
            # Compute x,y bounding box coordinates for object
            # Update bounding box rectangles list
            box = detections[0, 0, i, 3:7] * np.array([W,H,W,H])
            rectangles.append(box.astype("int"))

            # Draw bounding box around the object
            (start_x, start_y, end_x, end_y) = box.astype("int")
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,255,0), 1)

    # Update centroid tracker with computed bounding boxes
    objects = ct.update(rectangles)
    return objects


