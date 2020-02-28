

#  Stores information about the trackable objects
# @param object_id: Unique ID of the tracked object
# @param centroid: Previous centroid location of the tracked object
class TrackableObject():
    def __init__(self, object_id, centroid):
        # Store object ID and initialize a list of centroids
        # Use current centroid
        self.object_id = object_id
        self.centroid = [centroid]

        # Bool to indicate if a object has already been counted
        self.counted = False
