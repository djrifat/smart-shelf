from threading import Thread
import cv2

'''
Class to handle the capture of a videostream provided by a usb webcam
'''
class WebcamVideoCapture:

    def __init__(self, src=0):
        # Init video capture and read first frame
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        # Indicate if thread should stop
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        while True:
            # Stop thread
            if self.stopped:
                return
            # Otherwise read next frame
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return most recent frame
        return self.frame
    
    def stop(self):
        self.stopped = True

    def set_buffer(self):
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
