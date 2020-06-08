# Created by @DJrif

'''
Detect objects using centroid tracking method. 
Centroids are calculated by there Euclidean distance.
The tracker takes any object detector as input, 
provided it produces an set of bounding boxes ((x1,y1) (x2,y2))
'''

# Import necessary packages
import numpy as np
import cv2
from scipy.spatial import distance as dist
from collections import OrderedDict
from imutils.video import FPS

class CentroidTracker():
	'''
	Class method tracks objects using their centroid (center point),
	the centroid is updated each frame by calculating the Euclidean distance 
	between the existing and new centroid. 
	'''

	# @param maxDisappeared: Number of consecutive frames an object is allowed to "disappear"
	def __init__(self, maxDisappeared=50, maxDistance=50):
		# Initialize unique object ID
		# Ordered dictionaries to keep track of object ID's and its centroids
		self.next_object_ID = 0
		self.total_persons_detected = 0
		self.fps = FPS()
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.maxDisappeared = maxDisappeared
		self.maxDistance = maxDistance

	# Add new objects to tracker (object dictionary)
	# @param centroid:
	def register(self, centroid):
		# Use next available object ID to store the centroid
		self.objects[self.next_object_ID] = centroid
		self.disappeared[self.next_object_ID] = 0
		print("[INFO] OBJECT DETECTED")
		print("[INFO] OBJECT REGISTERED WITH ID: ", self.next_object_ID)
		print("[INFO] START TRACKING DWELL TIME OF ID: ", self.next_object_ID)
		self.next_object_ID += 1
		self.total_persons_detected += 1
		self.fps.start()
		#print("[INFO] OBJECT DETECTED")
		#print("[INFO] OBJECT REGISTERED WITH ID: ", self.next_object_ID)
	
	def total_detections(self):
		print("[INFO] TOTAL DETECTIONS TODAY: ", self.total_persons_detected)

	# Deregister objects from the tracker
	# @param object_ID:
	def deregister(self, object_ID):
		# Delete object ID from respective dictionaries
		# to deregister an object ID
		del self.objects[object_ID]
		del self.disappeared[object_ID]
		self.fps.stop()
		print("[INFO] OBJECT DEREGISTERED WITH ID: ", object_ID)
		print("[INFO] TIME DETECTED: {:.2f}".format(self.fps.elapsed()), "ID: ", object_ID)
		self.next_object_ID -= 1

	def time_alive(self, object_ID):
		return None

	# Update centroid tracker
	# @param rectangles: List of bounding box rectangles, from an object detector. Input format tulpe(startX, startY, endX, endY)
	# @return self.objects: Object list
	def update(self, rectangles):
		# Check if bounding box input list is empty
		if len(rectangles) == 0:
			# Loop over existing tracked objects and mark them as disappeared
			for object_ID in list(self.disappeared.keys()):
				self.disappeared[object_ID] += 1

				# Check if max number consecutive frames
				# is reached for a given object and remove if needed
				if self.disappeared[object_ID] > self.maxDisappeared:
					self.deregister(object_ID)

			# Return early if there's no object tracking info
			return self.objects

		# Initialize array of input centroids for the current frame
		input_centroids = np.zeros((len(rectangles), 2), dtype="int")

		# Loop through bounding boxes
		for (i, (startX, startY, endX, endY)) in enumerate(rectangles):
			# Derive the centroid using the bounding box coordinates
			# Store derived coordinates in numpy array
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			input_centroids[i] = (cX, cY)

		# If no objects are tracked
		# take input centroids and register them
		if len(self.objects) == 0:
			for i in range(0, len(input_centroids)):
				self.register(input_centroids[i])

		# If objects are being tracked
		# match the input centroids to existing object centroids
		else:
			# Fetch set of objects IDs and corresponding centroids
			object_IDs = list(self.objects.keys())
			object_centroids = list(self.objects.values())

			# Compute distance (Euclidean) between object centroid and input centroid
			# to match an input centroid to an existing object centroid
			# OUTPUT: Numpy array of shape (# of object centroids, # of input centroids)
			object_centroid_distance = dist.cdist(np.array(object_centroids), input_centroids)

			# Find smallest value in each row, sort row index based on minimal value
			# Sets the row with the smallest value at the front of the index list
			rows = object_centroid_distance.min(axis=1).argsort()

			# Find smallest value in each column
			# and sort based on previously computed row index
			cols = object_centroid_distance.argmin(axis=1)[rows]

			# Track rows and columns indexes that have been examined
			# in order to determine what action needs to be taken (update, register, deregister)
			used_rows = set()
			used_cols = set()
			
			for (row, col) in zip(rows, cols):

				# Ignore if already examined
				if row in used_rows or col in used_cols:
					continue
				
				# Check if centroid distance preceeds maximum
				if object_centroid_distance[row, col] > self.maxDistance:
					continue

				# grab object ID(current row) set new centroid if not examined
				object_ID = object_IDs[row]
				self.objects[object_ID] = input_centroids[col]
				self.disappeared[object_ID] = 0

				# Indicates that each row/column index are examined
				used_rows.add(row)
				used_cols.add(col)

			# Compute unexamined row and col index 
			unused_rows = set(range(0, object_centroid_distance.shape[0])).difference(used_rows)
			unused_cols = set(range(0, object_centroid_distance.shape[1])).difference(used_cols)

			# If object centroids(current) are bigger then input centroids(new),
			# check for lost or disappeared objects
			if object_centroid_distance.shape[0] >= object_centroid_distance.shape[1]:					
				for row in unused_rows:
					
					# Grab id and increment counter
					object_ID = object_IDs[row]
					self.disappeared[object_ID] += 1

					# Check if object exceeds max amount of consecutive frames
					# it's allowed to disppear. If so, deregister the object
					if self.disappeared[object_ID] > self.maxDisappeared:
						self.deregister(object_ID)

			# Nr of input centroids(new) is greater existing centroids,
			# register new input centroids 
			else:
				for col in unused_cols:
					self.register(input_centroids[col])

		self.fps.update()
		# Return tracked objects
		return self.objects



						

					



