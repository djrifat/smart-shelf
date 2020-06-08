# USAGE
# python object_tracker.py --conf utils/config.json

#from trackable_object import TrackableObject
from centroidtracker import CentroidTracker
from imutils.video import VideoStream, FPS
from utils.conf import Conf
import operator
import numpy as np
import argparse
import imutils
import time
import requests
import PIL
from PIL import Image
from io import BytesIO, StringIO
import cv2

# Construct argument parser and parse argumentsa()
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--conf', required=True,
    help='Path to config file')
args = vars(ap.parse_args())
conf = Conf(args["conf"])

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
    
    return parsed_response
    

# Initialize centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W)= (None, None)

# Load serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(conf["prototxt"], conf["model"])

# Initialize video stream and warmup camera sensor
print("[INFO] starting video stream")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Start FPS counter
# and initialize total number of processed frames
total_frames = 0
fps = FPS().start()

# Loop through frames from video stream
while True:
    # Read frame and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=conf["frame_width"])

    # Grab frames if dimensions are None
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Create a blob from the frame, pass the frame through 
    # the CNN to obtain predections and initialize list of bounding box rectangles
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W,H), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rectangles = []
    response = make_request(frame)

    # Process detections
    # Loop through detections
    for i in range(0, detections.shape[2]):
        # Filter out weak detections
        # Ensure predicted probability is greater then minimum threshold
        if detections[0, 0, i ,2] > conf['confidence']:
            # Compute x,y bounding box coordinates for object
            # Update bounding box rectangles list
            box = detections[0, 0, i, 3:7] * np.array([W,H,W,H])
            rectangles.append(box.astype("int"))
            
            # Draw bounding box around the object
            (start_x, start_y, end_x, end_y) = box.astype("int")
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,255,0), 1)

            # Confidence check face detection
            confidence = detections[0, 0 ,i ,2]

    # Update centroid tracker with computed bounding boxes
    objects = ct.update(rectangles)

    
    for face in response:

        face_attribute = face['faceAttributes']
        #face_rect = face['faceRectangle']
        emotions = face['faceAttributes']['emotion']
        current_mood = max(emotions.items(), key=operator.itemgetter(1))[0]
        #left, top, width, height = face_rect['left'], face_rect['top'], face_rect['width'], face_rect['height']
        #(start_x, start_y, end_x, end_y) = rectangles

        face_display = {
            'gender': face_attribute['gender'],
            'age': face_attribute['age'],
            'mood': current_mood
        }


    for k,v in face_display:
        print(k,v)

    # Loop through tracked objects
    for (object_ID, centroid) in objects.items():

        # Draw ID and centroid of the object in the output frame
        text = "ID {}".format(object_ID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("Foo", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Update FPS count
    total_frames += 1
    fps.update()

# Stop FPS count and display the varbiables
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()











