# Created by @DJrif

# Import necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

# TODO
# Assign unique ID to each detected object
# Keep track of objects and decide when to deregister them
	# Loop through object list to append new id's
	# and to check if an object needs to be deregistered
	#
# Use Eulclidean distance to calculate new centroid of an tracked object
	# Store and update in ordered dictionary



class CentroidTracker():

	# @param maxDisappeared:
	def __init__(self, maxDisappeared=50):
		# Initialize unique object ID
		# Ordered dictionaries to keep track of object ID's and its centroids
		self.next_object_ID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		# Number of consecutive frames an object is allowed
		# to be marked as "disappeared" untill it needs to be deregistered from tracking
		self.maxDisappeared = maxDisappeared


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

	# summary
	# @param rectangles:
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








