# python track.py --conf utils/config.json

from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType
from azure.cognitiveservices.vision.face.models import APIErrorException, DetectedFace
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face import FaceClient
from centroidtracker import CentroidTracker
from imutils.video import VideoStream, FPS
from io import BytesIO, StringIO
from utils.conf import Conf
from PIL import Image
import multiprocessing
import numpy as np
import matplotlib
import threading
import requests
import argparse
import imutils
import time
import dlib
import cv2
import PIL
import os
'''
# Construct argument parser and parse argumentsa()
"""
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--conf', required=True,
    help='Path to config file')
args = vars(ap.parse_args())
conf = Conf(args["conf"])
"""

ENDPOINT = 'https://eastus.api.cognitive.microsoft.com'
KEY = '2d0523e810c24bd5b7fd4448fbf71c67'
#KEY = '5d4e91c5581544229d0cdc2bc73d89e1'   # Mirabeau
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

def get_rectangle(face_dict):
    rect = face_dict['faceRectangle']
    left = rect['left']
    top = rect['top']
    right = left + rect['width']
    bottom = top + rect['height']
    
    return ((left, top), (right, bottom))


def get_faces(frame):
    return None

def get_emotions(frame):
    return None



total_number_of_faces = 0
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
video_capture = cv2.VideoCapture(0)

while True:
    video_frame_captured, video_frame = video_capture.read()
    video_frame = imutils.resize(video_frame, width=400)

    if video_frame_captured == True:
        #gray_video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        #gray_video_frame = cv2.equalizeHist(gray_video_frame)
        faces = face_cascade.detectMultiScale(video_frame)
        faces_in_frame = len(faces)

        print(f'number of faces in the frame: {total_number_of_faces}')

        if faces_in_frame != total_number_of_faces:
            total_number_of_faces = faces_in_frame
            print("change")
            
            if total_number_of_faces > 0:
                retval, video_frame_buffer = cv2.imencode(".jpg", video_frame)

                

    cv2.imshow("Mirabeau smart shelf", video_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
'''

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--conf', required=True,
    help='Path to config file')
args = vars(ap.parse_args())
conf = Conf(args["conf"])

detect_label = "Detected"
follow_label = "Following"
label = "label"

def start_tracker(box, label, rgb, input_queue, output_queue):
	# construct a dlib rectangle object from the bounding box
	# coordinates and then start the correlation tracker
	t = dlib.correlation_tracker()
	rect = dlib.rectangle(box[0], box[1], box[2], box[3])
	t.start_track(rgb, rect)

	# loop indefinitely -- this function will be called as a daemon
	# process so we don't need to worry about joining it
	while True:
		# attempt to grab the next frame from the input queue
		rgb = input_queue.get()

		# if there was an entry in our queue, process it
		if rgb is not None:
			# update the tracker and grab the position of the tracked
			# object
			t.update(rgb)
			pos = t.get_position()

			# unpack the position object
			start_x = int(pos.left())
			start_y = int(pos.top())
			end_x = int(pos.right())
			end_y = int(pos.bottom())

			# add the label + bounding box coordinates to the output
			# queue
			output_queue.put((label, (start_x, start_y, end_x, end_y)))


input_queues = []
output_queues = []

# Load serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(conf["prototxt"], conf["model"])

# Initialize video stream and warmup camera sensor
print("[INFO] starting video stream")
vs = VideoStream(src=0).start()
#time.sleep(2.0)

# start the frames per second throughput estimator
total_frames = 0
fps = FPS().start()

while True:

    # Read frame and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame is None:
        break
    
    if len(input_queues) == 0:

        (H,W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (W,H), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]
            if detections[0, 0, i ,2] > conf['confidence']:

                box = detections[0, 0, i, 3:7] * np.array([W,H,W,H])
                print(box)
                (start_x, start_y, end_x, end_y) = box.astype("int")
                bb = (start_x, start_y, end_x, end_y)

                iq = multiprocessing.Queue()
                oq = multiprocessing.Queue()
                input_queues.append(iq)
                output_queues.append(oq)

                # spawn a daemon process for a new object tracker
                p = multiprocessing.Process(
                    target=start_tracker,
                    args=(bb, label, rgb, iq, oq))
                p.daemon = True
                p.start()

                # Draw rectangle provided by the object detector
                cv2.rectangle(frame, (start_x,start_y),(end_x,end_y),
					(0, 255, 0), 2)
                print("drawn")
                cv2.putText(frame, detect_label, (start_x, start_y - 15),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    else:

        for iq in input_queues:
            iq.put(rgb)

        for oq in output_queues:
            (label, (start_x, start_y, end_x, end_y)) = oq.get()
            # Draw rectangle provided by the object tracker
            cv2.rectangle(frame, (start_x,start_y),(end_x,end_y),
                (0, 255, 0), 2)
            cv2.putText(frame, follow_label, (start_x, start_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)


        # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
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


