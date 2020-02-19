

class TrackableObject(self, object_id, centroid):
    # Store object ID and initialize a list of centroids
    # Use current centroid
    self.object_id = object_id
    self.centroid = [centroid]

    # Bool to indicate if a object has already been counted
    self.counted = False
