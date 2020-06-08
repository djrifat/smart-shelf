# USAGE
# python object_tracker.py --conf utils/config.json

from utils.detector_utils import detect_faces
from centroidtracker import CentroidTracker
from imutils.video import VideoStream, FPS
from utils.conf import Conf
import numpy as np
import argparse
import imutils
import sched
import time
import cv2

# Construct argument parser and parse argumentsa()
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--conf', required=True,
    help='Path to config file') 
args = vars(ap.parse_args())
conf = Conf(args["conf"])

# Initialize centroid tracker
ct = CentroidTracker()

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
    frame = imutils.resize(frame, width=conf["frame_width"])
    frame = cv2.flip(frame, 1)
    objects = detect_faces(frame, net ,ct)

    # Loop through tracked objects
    for (object_ID, centroid) in objects.items():       
        # Draw ID and centroid of the object in the output frame
        text = "ID {}".format(object_ID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)

    cv2.imshow("Mirabeau smart shelf", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Update FPS count
    fps.update()

# Stop FPS count and display the varbiables
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
ct.total_detections()

cv2.destroyAllWindows()
vs.stop()

