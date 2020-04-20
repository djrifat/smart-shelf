# Created by @DJrif

'''
Detect objects using centroid tracking method. 
Centroids are calculated by there Euclidean distance.
The tracker takes any object detector as input, 
provided it produces an set of bounding boxes ((x1,y1) (x2,y2))
'''

# Import necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
from imutils.video import FPS
import numpy as np



class CentroidTracker():
	'''
	Class method tracks objects using their centroid (center point),
	the centroid is updated each frame by calculating the Euclidean distance 
	between the existing and new centroid. 
	'''

	# @param max_disappeared: Number of consecutive frames an object is allowed to "disappear"
	def __init__(self, max_disappeared=50, max_distance=50):
		# Initialize unique object ID
		# Ordered dictionaries to keep track of object ID's and its centroids
		self.next_object_ID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		# Number of consecutive frames an object is allowed
		# to be marked as "disappeared" untill it needs to be deregistered from tracking
		self.max_disappeared = max_disappeared
		self.max_distance = max_disappeared

	# Add new objects to tracker (object dictionary)
	# @param centroid:
	def register(self, centroid):
		# Use next available object ID to store the centroid
		self.objects[self.next_object_ID] = centroid
		self.disappeared[self.next_object_ID] = 0
		self.next_object_ID += 1

	# Deregister objects from the tracker
	# @param object_ID:
	def deregister(self, object_ID):
		# Delete object ID from respective dictionaries
		# to deregister an object ID
		del self.objects[object_ID]
		del self.disappeared[object_ID]

	# Update centroid tracker
	# @param rectangles: List of bounding box rectangles, from an object detector. Input format tulpe(startX, startY, endX, endY)
	# @return self.objects: 
	def update(self, rectangles):
		# Check if bounding box input list is empty
		if len(rectangles) == 0:
			# Loop over existing tracked objects and mark them as disappeared
			for object_ID in list(self.disappeared.keys()):
				self.disappeared[object_ID] += 1

				# Check if max number consecutive frames
				# is reached for a given object and remove if needed
				if self.disappeared[object_ID] > self.max_disappeared:
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

			# Find smallest value in each row
			# Sort tow index based on minimal value
			# Sets the row with the smallest value at the front of the index list
			rows = object_centroid_distance.min(axis=1).argsort()

			# Find smallest value in each column
			# and sort based on previously computed row index
			cols = object_centroid_distance.argmin(axis=1)[rows]

			# Track rows and columns indexes that have been examined
			# in order to determine what action needs to be taken (update, register, deregister)
			used_rows = set()
			used_cols = set()
			
			# Loop over row, column index tulpes
			for (row, col) in zip(rows, cols):

				# If value (row/col) already has been examined ignore it
				if row in used_rows or col in used_cols:
					continue

				if object_centroid_distance[row,col] > self.max_distance:
					continue

				# if value hasn't been examined yet,
				# grab object ID(current row) set new centroid
				# reset dissapeared counter
				object_ID = object_IDs[row]
				self.objects[object_ID] = input_centroids[col]
				self.disappeared[object_ID] = 0

				# Indicates that each row/column index 
				# have been examined, respectively
				used_rows.add(row)
				used_cols.add(col)

			# Compute row and col index that haven't been examined yet
			unused_rows = set(range(0, object_centroid_distance.shape[0])).difference(used_rows)
			unused_cols = set(range(0, object_centroid_distance.shape[1])).difference(used_cols)

			# In case object centroids(current) are bigger then the input centroids(new),
			# check if any objects are lost or disappeared
			if object_centroid_distance.shape[0] >= object_centroid_distance.shape[1]:					
				for row in unused_rows:
					# Take object ID for matching row index 
					# and increment disappeared counter
					object_ID = object_IDs[row]
					self.disappeared[object_ID] += 1

					# Check if object exceeds max amount of consecutive frames
					# it's allowed to disppear. If so, deregister the object
					if self.disappeared[object_ID] > self.max_disappeared:
						self.deregister(object_ID)

			# Nr of input centroids(new) is greater existing centroids,
			# register new input centroids 
			else:
				for col in unused_cols:
					self.register(input_centroids[col])

		# Return tracked objects
		return self.objects



						

					



