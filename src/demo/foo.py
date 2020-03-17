# USAGE
# python demo.py --conf utils/config.json

#from trackable_object import TrackableObject
from utils import detector_utils as detector_utils
from utils.detector_utils import WebcamVideoStream
from centroidtracker import CentroidTracker
from imutils.video import VideoStream, FPS
from multiprocessing import Queue, Pool
from utils.conf import Conf
import tensorflow as tf
import multiprocessing
import numpy as np
import argparse
import datetime
import imutils
import time
import cv2

# Initialize centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W)= (None, None)
frame_processed = 0
score_thresh = 0.2

# Create a worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue
def worker(input_q, output_q, cap_params, frame_processed):
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.compat.v1.Session(graph=detection_graph)
    while True:
        frame = input_q.get()
        if (frame is not None):
            # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)
            # draw bounding boxes
            detector_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                frame)
            # add frame annotated with bounding box to queue
            output_q.put(frame)
            frame_processed += 1
        else:
            output_q.put(frame)
    sess.close()
# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-src',
    '--source',
    dest='video_source',
    type=int,
    default=0,
    help='Device index of the camera.')
parser.add_argument(
    '-nhands',
    '--num_hands',
    dest='num_hands',
    type=int,
    default=20,
    help='Max number of hands to detect.')
parser.add_argument(
    '-fps',
    '--fps',
    dest='fps',
    type=int,
    default=1,
    help='Show FPS on detection/display visualization')
parser.add_argument(
    '-wd',
    '--width',
    dest='width',
    type=int,
    default=300,
    help='Width of the frames in the video stream.')
parser.add_argument(
    '-ht',
    '--height',
    dest='height',
    type=int,
    default=300,
    help='Height of the frames in the video stream.')
parser.add_argument(
    '-ds',
    '--display',
    dest='display',
    type=int,
    default=1,
    help='Display the detected images using OpenCV. This reduces FPS')
parser.add_argument(
    '-num-w',
    '--num-workers',
    dest='num_workers',
    type=int,
    default=4,
    help='Number of workers.')
parser.add_argument(
    '-q-size',
    '--queue-size',
    dest='queue_size',
    type=int,
    default=5,
    help='Size of the queue.')
parser.add_argument(
    '-c',
    '--caffe,',
    dest='caffe',
    default="models/res10_300x300_ssd_iter_140000.caffemodel",
    help='Path to caffe model')
parser.add_argument(
    '-p',
    '--prototxt,',
    dest='prototxt',
    default="models/deploy.prototxt",
    help='Path to prototxt')
parser.add_argument(
    '-conf',
    '--confidence,',
    dest='confidence',
    type=float,
    default=0.6,
    help='Path to prototxt')
args = parser.parse_args()

# Load serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.caffe)

# Input, Output queue's
input_q = Queue(maxsize=args.queue_size)
output_q = Queue(maxsize=args.queue_size)

# Initialize video stream and warmup camera sensor
print("[INFO] starting video stream")
video_capture = WebcamVideoStream(
    src=args.video_source, width=args.width, height=args.height).start()
time.sleep(2.0)

cap_params = {}
frame_processed = 0
cap_params['im_width'], cap_params['im_height'] = video_capture.size()
cap_params['score_thresh'] = score_thresh

# max number of hands we want to detect/track
cap_params['num_hands_detect'] = args.num_hands

# spin up workers to paralleize detection.
pool = Pool(args.num_workers, worker,
            (input_q, output_q, cap_params, frame_processed))

# Start FPS counter
# and initialize total number of processed frames
total_frames = 0
fps = FPS().start()

cv2.namedWindow("Mirabeau Smart Shelf", cv2.WINDOW_NORMAL)
# Loop through frames from video stream
while True:
    # Read frame and resize it
    #hand_frame = video_capture.read()
    face_frame = video_capture.read()
    #hand_frame = imutils.resize(hand_frame, width=args.width)
    #face_frame = imutils.resize(face_frame, width=args.width)

    # Grab frames if dimensions are None
    if W is None or H is None:
        (H, W) = face_frame.shape[:2]
        #(H, W) = hand_frame.shape[:2]

    # Create a blob from the frame, pass the frame through 
    # the CNN to obtain predections and initialize list of bounding box rectangles
    blob = cv2.dnn.blobFromImage(face_frame, 1.0, (W,H), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rectangles = []

    # Process detections
    # Loop through detections
    for i in range(0, detections.shape[2]):
        # Filter out weak detections
        # Ensure predicted probability is greater then minimum threshold
        if detections[0, 0, i ,2] > args.confidence:
            # Compute x,y bounding box coordinates for object
            # Update bounding box rectangles list
            box = detections[0, 0, i, 3:7] * np.array([W,H,W,H])
            rectangles.append(box.astype("int"))
            
            # Draw bounding box around the object
            (start_x, start_y, end_x, end_y) = box.astype("int")
            cv2.rectangle(face_frame, (start_x, start_y), (end_x, end_y), (0,255,0), 2)

            # Confidence check face detection
            confidence = detections[0, 0 ,i ,2]

    # Update centroid tracker with computed bounding boxes
    objects = ct.update(rectangles)

    # Loop through tracked objects
    for (object_ID, centroid) in objects.items():

        # Draw ID and centroid of the object in the output frame
        text = "ID {}".format(object_ID)
        text2 = "{:.2f}%".format(confidence * 100)
        cv2.putText(face_frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(face_frame, text2, (start_x, start_y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.circle(face_frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)

    input_q.put(cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB))
    output_frame = output_q.get()
    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

    #final = cv2.hconcat([face_frame,output_frame])
    # Show output frame
    if (output_frame is not None):
        if (args.display > 0):
            cv2.imshow("Mirabeau Smart Shelf", output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break

    # Update FPS count
    total_frames += 1
    fps.update()

# Stop FPS count and display the varbiables
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

pool.terminate()
cv2.destroyAllWindows()
video_capture.stop()


