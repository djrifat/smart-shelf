# USAGE
# python demo.py --conf utils/config.json
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
api_call_threshold = 2
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

    try:
        (H,W) = utils.detector_utils.grab_frame_dim(frame)
    except AttributeError:
        print("Shape not found")

    status = "Waiting..."
    rectangles = []

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

                (start_x, start_y, end_x, end_y) = box.astype("int")
                t = dlib.correlation_tracker()
                rect = dlib.rectangle(start_x, start_y, end_x, end_y)
                t.start_track(rgb, rect)

                trackers.append(t)

        faces_in_frame = len(trackers)
        if faces_in_frame != total_faces:
            total_faces = faces_in_frame

        if total_faces >= api_call_threshold:
            print("Threshold reached sending API request")
            try:
                response = utils.detector_utils.make_request(frame)
            except ValueError as e:
                print("Some error occurred: ", e)

            if not response:
                print("Nothing detected")
                test, emotion = {},{}
            else:
                test = response[0]
                emotion = test['faceAttributes']['emotion']
                
            #print("-----: ",emotion)
            #print(type(emotion))
  
    else:  
        for t in trackers:          
            status = "Tracking..."
            utils.detector_utils.unpack_tracker(frame, t, rgb, rectangles)
            #(start_X, start_Y, end_X, end_Y) = utils.detector_utils.unpack_rect(frame, t, rgb, rectangles)

            #for i, k  in enumerate(face_display):
                #cv2.putText(frame, "{0}: {1}".format(k, face_display[k]), 
                    #(left+width+5, top + 5 + 20*i),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                    #(start_x + (end_y-start_y)-25, start_y + 5 + 20*i),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    '''
    for face in response:
        
        face_attribute = face['faceAttributes']
        #face_rect = face['faceRectangle']
        emotions = face['faceAttributes']['emotion']
        current_mood = max(emotions.items(), key=operator.itemgetter(1))[0]
        #left, top, width, height = face_rect['left'], face_rect['top'], face_rect['width'], face_rect['height']
        response2 = dict(response)
        print(type(response2))
        face_display = {
            'gender': face_attribute['gender'],
            'age': face_attribute['age'],
            'mood': current_mood
        }   
    ''' 

    # Update centroid tracker with computed bounding boxes
    objects = ct.update(rectangles)

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

cap.stop()
cv2.destroyAllWindows()



