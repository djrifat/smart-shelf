# USAGE
# python object_tracking.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel
# python object_tracking.py --conf utils/config.json

# import the necessary packages
from imutils.video import FPS, VideoStream
from utils import conf
import multiprocessing
import numpy as np
import argparse
import imutils
import dlib
import cv2

# Function to call for new process (Deamon process)
# @param box: Bouding box coordinates of the tracked object
# @param label: Label describing the object
# @param rb: RGB-ordered image to start the initial dlib object tracker
# @param input_queue: 
# @param output_queue: 
def start_tracker(box, label, rgb, input_queue, output_queue):
	# Construct dlib rectangle from bounding box
	# Coordinates and start the correlation tracker
	t = dlib.correlation_tracker()
	rect = dlib.rectangle(box[0], box[1], box[2], box[3])
	t.start_track(rgb, rect)

	# Function will be called as a daemon process
	while True:
		# grab next frame from input queue
		rgb = input_queue.get()

		# Update tracker, get tracked object position
		if rgb is not None:
			# Update tracker, get tracked object position
			t.update(rgb)
			pos = t.get_position()

			# unpack position object
			start_x = int(pos.left())
			start_y = int(pos.top())
			end_x = int(pos.right())
			end_y = int(pos.bottom())

			# add the label + bounding box coordinates
			output_queue.put((label, (start_x, start_y, end_x, end_y)))

# Construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--conf', required=True,
    help='Path to config file')
args = vars(ap.parse_args())
conf = conf.Conf(args["conf"])

# Initialize list of input -and output Queue's
# Will hold objects that are being tracked
input_queues = []
output_queues = []

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# Load serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(conf["prototxt"], conf["model"])

# initialize the video stream and output video writer
print("[INFO] starting video stream...")
#vs = cv2.VideoCapture(args["video"])
vs = VideoStream(src=0).start()

# start the frames per second throughput estimator
fps = FPS().start()

# Loop through frames from video stream
while True:
	# Grab next frame
	#(grabbed, frame) = vs.read()
	frame = vs.read()

	# Check if end of video is reached
	if frame is None:
		break

	# Resize frame --> higher FPS
	# Change color space to RGB (dlib uses BGR)
	frame = imutils.resize(frame, width=600)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Check if detection queue is empty
	# If queue is empty, no object tracker has been created
	if len(input_queues) == 0:
		# Create a blob from the frame dimensions, pass the frame through 
		# the CNN to obtain predections and initialize list of bounding box rectangles
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
		#blob = cv2.dnn.blobFromImage(frame, 1.0, (w,h), (104.0, 177.0, 123.0))
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# filter out weak detections by requiring a minimum
			# confidence
			confidence = detections[0, 0, i, 2]
			if confidence > conf["confidence"]:
				# extract the index of the class label from the
				# detections list
				id_x = int(detections[0, 0, i, 1])
				label = CLASSES[id_x]

				# Ignore if not a person
				#if CLASSES[id_x] != "person":
				#	continue

				# Compute x,y bounding box coordinates for object
				# Update bounding box rectangles list
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(start_x, start_y, end_x, end_y) = box.astype("int")
				bb = (start_x, start_y, end_x, end_y)

				# create two brand new input and output queues,
				# respectively
				in_queue = multiprocessing.Queue()
				out_queue = multiprocessing.Queue()
				input_queues.append(in_queue)
				output_queues.append(out_queue)

				# Spawn daemon process for object tracker
				p = multiprocessing.Process(
					target=start_tracker,
					args=(bb, label, rgb, in_queue, out_queue))
				p.daemon = True
				p.start()

				# Draw bounding box and corresponding label
				cv2.rectangle(frame, (start_x, start_y), (end_x, end_y),
					(0, 255, 0), 2)
				cv2.putText(frame, label, (start_x, start_y - 15),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# If detection already has been performed, start tracking multiple objects
	else:
		# Loop through all input queues to add RGB frame
		# Updates each object tracker running on seperate thread 
		for in_queue in input_queues:
			in_queue.put(rgb)

		# Loop over each of the output queues
		for out_queue in output_queues:
			# grab the updated bounding box coordinates for the
			# object -- the .get method is a blocking operation so
			# this will pause our execution until the respective
			# process finishes the tracking update
			(label, (start_x, start_y, end_x, end_y)) = out_queue.get()

			# Draw the bounding box from the correlation object tracker
			cv2.rectangle(frame, (start_x, start_y), (end_x, end_y),
				(0, 255, 0), 2)
			cv2.putText(frame, label, (start_x, start_y - 15),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)


	# Show output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


# Cleanup
cv2.destroyAllWindows()
vs.stop()
