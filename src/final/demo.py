# USAGE
# python demo.py --conf utils/config.json
# python3 demo.py --conf utils/config.json
from utils.WebcamVideoCapture import WebcamVideoCapture
from centroidtracker import CentroidTracker
from imutils.video import VideoStream, FPS
from utils.conf import Conf
import utils.detector_utils
import numpy as np
import argparse
import operator
import imutils
import sched
import json
import time
import dlib
import cv2

# Construct argument parser and parse argumentsa()
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--conf', required=True,
    help='Path to config file')
args = vars(ap.parse_args())
conf = Conf(args["conf"])

# Initialize centroid tracker and nessecary parameters
ct = CentroidTracker()
#skip_frames = conf["skip_frames"]
skip_frames = 30
trackers = []
total_frames = 0

total_faces = 0
faces_in_frame = 0
api_call_threshold = 1
frame_buffer_size = 2

# Load serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNet(conf["prototxt"], conf["model"])

# Initialize video stream and warmup camera sensor and start FPS counter
print("[INFO] starting video stream")
cap = WebcamVideoCapture(src=0).start()
time.sleep(2.0)
fps = FPS().start()  

# Loop through frames from video stream
while True:
    # Read frame and resize it
    frame = cap.read()
    cap.set_buffer(frame_buffer_size)
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

    # Grab frame dimensions
    try:
        (H,W) = utils.detector_utils.grab_frame_dim(frame)
    except AttributeError:
        print("Shape not found")

    status = "Waiting..."
    rectangles = []

    # Perform detection every N fames
    if total_frames % skip_frames == 0:

        status = "Detecting..."
        trackers = []

        blob = cv2.dnn.blobFromImage(frame, 1.0, (W,H), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # Loop through detections
        for i in np.arange(0, detections.shape[2]):
            # Filter out weak detections
            if detections[0, 0, i ,2] > conf['confidence']:
                # Compute x,y bounding box coordinates for object
                # Update bounding box rectangles list
                box = detections[0, 0, i, 3:7] * np.array([W,H,W,H])

                # Unpack boundingbox coordinates and append to dlib tracker
                (start_x, start_y, end_x, end_y) = box.astype("int")
                t = dlib.correlation_tracker()
                rect = dlib.rectangle(start_x, start_y, end_x, end_y)
                t.start_track(rgb, rect)

                trackers.append(t)

        # Keep track of detected faces
        faces_in_frame = len(trackers)
        if faces_in_frame != total_faces:
            total_faces = faces_in_frame

        # Make API call on condition
        if total_faces >= api_call_threshold:
            print("[INFO] Threshold reached sending API request")
            try:
                response = utils.detector_utils.make_request(frame)
            except ValueError as e:
                print("Some error occurred: ", e)

            # Handle empty response
            if not response:
                print("[INFO] Nothing detected")
                emotion = {}
            else:
                print("[INFO] API call succesful")
                #for face in response:
                #for i in response:
                    #for x in i:
                        #print("TEST", i[x])
                #for i in response:
                    #emotion = response[0]['faceAttributes']['emotion'] 
                    #print("----------",emotion)  
        else:
            response = []
  
    else:  
        for t in trackers:          
            status = "Tracking..."
            utils.detector_utils.unpack_tracker(frame, t, rgb, rectangles)

    # Update centroid tracker with computed bounding boxes
    objects = ct.update(rectangles)

    # Loop through API response
    

    # Loop through tracked objects
    for (object_ID, centroid) in objects.items():   
        # Draw ID and centroid of the object in the output frame
        text = "ID {}".format(object_ID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        for face in response:

            face_attribute = face['faceAttributes']
            face_rect = face['faceRectangle']
            emotions = face['faceAttributes']['emotion']
            current_mood = max(emotions.items(), key=operator.itemgetter(1))[0]

            left, top, width, height = face_rect['left'], face_rect['top'], face_rect['width'], face_rect['height']

            face_display = {
                'gender': face_attribute['gender'],
                'age': face_attribute['age'],
                'mood': current_mood
            } 

            # Display retrieved emotions from API call
            for i, k  in enumerate(face_display):
                cv2.putText(frame, "{0}: {1}".format(k, face_display[k]),
                    #(left+width+5, top + 5 + 20*i),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                    (centroid[0] + 15, centroid[1]+20*i),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
     
    # Visualize status information
    info = [("Status: ", status)]
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    # Display frames
    cv2.imshow("Mirabeau smart shelf", frame)

    # Exit program on key press
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

cap.stop()
cv2.destroyAllWindows()



