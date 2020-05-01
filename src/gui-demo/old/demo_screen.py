# USB camera display using PyQt and OpenCV, from iosoft.blog
# Copyright (c) Jeremy P Bentham 2019
# Please credit iosoft.blog if you use the information or software in it

# python demo_screen.py --conf utils/config.json

VERSION = "Mirabeau Smart Shelf Demo"

import sys, time, threading, cv2, argparse
from imutils.video import VideoStream, FPS
import numpy as np
from utils.conf import Conf
from utils.detector_utils import detect_faces


from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer, QPoint, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLabel
from PyQt5.QtWidgets import QWidget, QAction, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QFont, QPainter, QImage, QTextCursor
from utils.detector_utils import WebcamVideoStream
from centroidtracker import CentroidTracker
try:
    import Queue as Queue
except:
    import queue as Queue


# Construct argument parser and parse argumentsa()
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--conf', required=True,
    help='Path to config file')
args = vars(ap.parse_args())
conf = Conf(args["conf"])

IMG_SIZE    = 1280,720          # 640,480 or 1280,720 or 1920,1080
IMG_FORMAT  = QImage.Format_RGB888
DISP_SCALE  = 2                # Scaling factor for display image
DISP_MSEC   = 50                # Delay between display cycles
CAP_API     = cv2.CAP_ANY       # API: CAP_ANY or CAP_DSHOW etc...
EXPOSURE    = 0                 # Zero for automatic exposure
TEXT_FONT   = QFont("Courier", 10)

camera_num  = 1                 # Default camera (first in list)
image_queue = Queue.Queue()     # Queue to hold images
capturing   = True              # Flag to indicate capturing

# Grab images from the camera (separate thread)
def grab_images(cam_num, queue):
    cap = cv2.VideoCapture(cam_num-1 + CAP_API)
    while capturing:
        if cap.grab():
            retval, image = cap.retrieve(0)
            if image is not None and queue.qsize() < 2:
                queue.put(image)
            else:
                time.sleep(DISP_MSEC / 1000.0)
        else:
            print("Error: can't grab camera image")
            break
    cap.release()


# Initialize centroid tracker
ct = CentroidTracker()

# Load serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(conf["prototxt"], conf["model"])

# Initialize video stream and warmup camera sensor
print("[INFO] starting video stream")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Image widget
class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        self.setMinimumSize(image.size())
        self.update()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        qp.end()

# Main window
class MyWindow(QMainWindow):

    # Create main window
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

        self.central = QWidget(self)
        self.vlayout = QVBoxLayout()        # Window layout
        self.displays = QHBoxLayout()
        self.disp = ImageWidget(self)    
        self.displays.addWidget(self.disp)
        self.vlayout.addLayout(self.displays)
        self.label = QLabel(self)
        self.vlayout.addWidget(self.label)
        self.central.setLayout(self.vlayout)
        self.setCentralWidget(self.central)
        self.setWindowTitle(VERSION)

    # Start image capture & display
    def start(self):
        self.timer = QTimer(self)           # Timer to trigger display
        self.timer.timeout.connect(lambda: 
                    self.show_image(image_queue, self.disp, DISP_SCALE))
        self.timer.start(DISP_MSEC)         
        self.capture_thread = threading.Thread(target=grab_images, 
                    args=(camera_num, image_queue))
        self.capture_thread.start()         # Thread to grab images

    # Fetch camera image from queue, and display it
    def show_image(self, imageq, display, scale):
        if not imageq.empty():
            image = imageq.get()
            if image is not None and len(image) > 0:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.display_image(img, display, scale)

    # Display an image, reduce size if required
    def display_image(self, img, display, scale=1):
        disp_size = img.shape[1]//scale, img.shape[0]//scale
        disp_bpl = disp_size[0] * 3
        objects = detect_faces(img, net ,ct)
        if scale > 1:
            img = cv2.resize(img, disp_size, interpolation=cv2.INTER_CUBIC)
            #img = cv2.putText(img, 'OpenCV', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            #img = cv2.rectangle(img, (50, 40), (100, 100), (0,255,0), 1)
            for (object_ID, centroid) in objects.items():       
                # Draw ID and centroid of the object in the output frame
                text = "ID {}".format(object_ID)
                cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.circle(img, (centroid[0], centroid[1]), 4, (0,255,0), -1)
        qimg = QImage(img.data, disp_size[0], disp_size[1], disp_bpl, IMG_FORMAT)
        display.setImage(qimg)




    # Window is closing: stop video capture
    def closeEvent(self, event):
        global capturing
        capturing = False
        self.capture_thread.join()

    def flush(self):
        pass



app = QApplication(sys.argv)
win = MyWindow()
win.show()
win.start()
sys.exit(app.exec_())

