# USAGE
# python snel.py --conf utils/config.json

#from trackable_object import TrackableObject
#from utils.detector_utils import WebcamVideoStream
from centroidtracker import CentroidTracker
from imutils.video import VideoStream, FPS
from utils.detector_utils import detect_faces
from utils.conf import Conf
import numpy as np
import argparse
import imutils
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

# Load serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(conf["prototxt"], conf["model"])

# Initialize video stream and warmup camera sensor
print("[INFO] starting video stream")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Start FPS counter
fps = FPS().start()  
total_faces = 0
trackers = []
    

def grab_frame_dim(frame):
    (H, W) = (None, None)
    if H is None or W is None:
        (H, W) = frame.shape[:2]
    return (H,W)

tracker = dlib.correlation_tracker()

# Loop through frames from video stream
while True:
    # Read frame and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=conf["frame_width"])
    frame = cv2.flip(frame, 1)
    #objects = detect_faces(frame, net ,ct, conf)

    (H,W) = grab_frame_dim(frame)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W,H), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rectangles = []
    frame_faces = 0

    # Process detections
    # Loop through detections
    for i in np.arange(0, detections.shape[2]):
        # Filter out weak detections
        # Ensure predicted probability is greater then minimum threshold
        if detections[0, 0, i ,2] > conf['confidence']:
            # Compute x,y bounding box coordinates for object
            # Update bounding box rectangles list
            box = detections[0, 0, i, 3:7] * np.array([W,H,W,H])
            rectangles.append(box.astype("int"))
            frame_faces = len(rectangles)
        
            if frame_faces != total_faces:
                total_faces = frame_faces

                if total_faces > 0:
                    buffer_frame = cv2.imencode(".jpg", frame)


            # Draw bounding box around the object
                    (start_x, start_y, end_x, end_y) = box.astype("int")
                    rect = dlib.rectangle(start_x, start_y, end_x, end_y)
                    tracker.start_track(frame, rect)
                    trackers.append(tracker)
                    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,255,0), 1)
                    print("detection")
            
            else:
                
                for t in trackers:
                    t.update(frame)
                    pos = t.get_position()
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    cv2.rectangle(frame, (startX, startY), (endX, endY),
				        (0, 255, 0), 2)

                    print("tracking")
                

    # Update centroid tracker with computed bounding boxes
    objects = ct.update(rectangles)
    #(start_x, start_y), (end_x, end_y) = objects

    # Loop through tracked objects
    for (object_ID, centroid) in objects.items():       
        # Draw ID and centroid of the object in the output frame
        text = "ID {}".format(object_ID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
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

cv2.destroyAllWindows()
vs.stop()

