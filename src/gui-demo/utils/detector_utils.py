
from imutils.video import VideoStream, FPS
from threading import Thread
import numpy as np
import argparse
import imutils
import sched
import time
import cv2

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


