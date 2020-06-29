# USAGE
# python dlib_tracker.py --conf utils/config.json

from centroidtracker import CentroidTracker
from imutils.video import VideoStream, FPS
from utils.conf import Conf
import utils.detector_utils
import numpy as np
import argparse
import imutils
import emoji
import sched
import time
import dlib
import cv2

# Construct argument parser and parse argumentsa()
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--conf', required=True,
    help='Path to config file')
args = vars(ap.parse_args())
conf = Conf(args["conf"])

# Initialize centroid tracker
ct = CentroidTracker()
trackers = []
total_frames = 0
total_faces = 0
skip_frames = conf["skip_frames"]

# Load serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(conf["prototxt"], conf["model"])

# Initialize video stream and warmup camera sensor
print("[INFO] starting video stream")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Start FPS counter
fps = FPS().start()  

# Loop through frames from video stream
while True:
    # Read frame and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    (H,W) = utils.detector_utils.grab_frame_dim(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    status = "Waiting"
    rects = []

    if total_frames % skip_frames == 0:

        status = "Detecting..."
        trackers = []

        blob = cv2.dnn.blobFromImage(frame, 1.0, (W,H), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # Loop through detections
        for i in np.arange(0, detections.shape[2]):
            # Filter out weak detections
            # Ensure predicted probability is greater then minimum threshold
            if detections[0, 0, i ,2] > conf['confidence']:
                # Compute x,y bounding box coordinates for object
                # Update bounding box rectangles list
                box = detections[0, 0, i, 3:7] * np.array([W,H,W,H])
                (start_x, start_y, end_x, end_y) = box.astype("int")

                t = dlib.correlation_tracker()
                rect = dlib.rectangle(start_x, start_y, end_x, end_y)
                t.start_track(rgb, rect)

                trackers.append(t)
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,255,0), 1)    
    else:  
        for t in trackers:          
            status = "Tracking..."
            utils.detector_utils.unpack_tracker(frame, t, rgb, rects)
         
    # Update centroid tracker with computed bounding boxes
    objects = ct.update(rects)

    # Loop through tracked objects
    for (object_ID, centroid) in objects.items():   
        # Draw ID and centroid of the object in the output frame
        text = "ID {}".format(object_ID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    info = [("Status: ", status)]
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    #cv2.imshow('result.png', result_o)
    cv2.imshow("Mirabeau smart shelf", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Update FPS count
    total_frames += 1
    fps.update()

# Stop FPS count and display the varbiables
fps.stop()
ct.total_detections()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()

