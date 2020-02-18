# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

from centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# Construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Initialize centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

temp_proto_path = 'deploy.prototxt'
temp_model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
temp_confidence = 0.5

# Load serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Initialize video stream and warmup camera sensor
print("[INFO] starting video stream")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Loop through frames from video stream
while True:

    # Read frame and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=640)

    # Grab frames if dimensions are None
    if W is None or H is None:
        (H,W) = frame.shape[:2]

    # Create a blob from the frame, pass the frame through 
    # the CNN to obtain predections 
    # and initialize list of bounding box rectangles
    print("[INFO] computing object detections...")
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W,H), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rectangles = []

    # Process detections
    # Loop through detections
    for i in range(0, detections.shape[2]):
        # Filter out weak detections
        # Ensure predicted probability is greater then minimum threshold
        if detections[0, 0 ,i ,2] > args['confidence']:
            # Compute x,y bounding box coordinates for object
            # Update bounding box rectangles list
            box = detections[0, 0, i, 3:7] * np.array([W,H,W,H])
            rectangles.append(box.astype("int"))
            # Draw bounding box around the object
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)

            # Confidence check face detection
            confidence = detections[0, 0 ,i ,2]
            if confidence < args['confidence']:
                continue

    # Update centroid tracker with computed bounding boxes
    objects = ct.update(rectangles)

    # Loop through tracked objects
    for (object_ID, centroid) in objects.items():

        # Draw ID and centroid of the object in the output frame
        text = "ID {}".format(object_ID)
        text2 = "{:.2f}%".format(confidence * 100)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(frame, text2, (startX, startY),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)

    # Show output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
