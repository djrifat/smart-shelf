
from utils.WebcamVideoCapture import WebcamVideoCapture
from centroidtracker import CentroidTracker
from imutils.video import VideoStream, FPS
from utils.conf import Conf
import utils.detector_utils
import numpy as np
import argparse
import operator
import imutils
import dlib
import time
import cv2

ct = CentroidTracker()
total_frames = 0
skip_frames = 30
trackers = []
face_display = {}
test_dict

total_faces = 0
faces_in_frame = 0
api_call_threshold = 2
frame_buffer_size = 2

print("[INFO] Starting video stream")
cap = WebcamVideoCapture(src=0).start()
time.sleep(2.0)
fps = FPS().start()

while True:

    frame = cap.read()
    cap.set_buffer(frame_buffer_size)
    frame = imutils.resize(frame, width=500)   
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    try:
        (H,W) = utils.detector_utils.grab_frame_dim(frame)
    except AttributeError:
        print("[ERROR] Shape not found")
    
    rectangles = []
    test_rect = []
    status = "Waiting..."
    
    if total_frames % skip_frames == 0:

        status = "Detecting..."
        trackers = []
        response = utils.detector_utils.make_request(frame)

        for face in response:
            face_attribute = face['faceAttributes']
            face_rect = face['faceRectangle']
            emotions = face['faceAttributes']['emotion']
            current_mood = max(emotions.items(), key=operator.itemgetter(1))[0]

            left, top, width, height = face_rect['left'], face_rect['top'], face_rect['width'], face_rect['height']
            (start_x, start_y), (end_x, end_y) = utils.detector_utils.get_rectangle(face)

            t = dlib.correlation_tracker()
            rect = dlib.rectangle(start_x,start_y,end_x,end_y)
            t.start_track(rgb, rect)
            trackers.append(t)
            
            face_display = {
                'gender': face_attribute['gender'],
                'age': face_attribute['age'],
                'mood': current_mood
            } 
            
            print("FACE DISPLAY", face_display, type(face_display))

        faces_in_frame = len(trackers)
        if faces_in_frame != total_faces:
            total_faces = faces_in_frame
        '''
        if total_faces >= api_call_threshold:
            print("[INFO] Faces in frame detected: ", total_faces)
            print("[INFO] Threshold reached sending API request")
            try:
                response = utils.detector_utils.make_request(frame)
            except ValueError as e:
                print("[ERROR] ", e)
            if not response:
                print("[INFO] Nothing detected")
            else:
                emotion = response[0]['faceAttributes']['emotion']  
                print("[INFO] Detected emotions: ", emotion)
        '''
    else:
        for t in trackers:
            status = "Tracking..."
            utils.detector_utils.unpack_tracker(frame, t, rgb, rectangles) 
            '''
            for face in response:

                face_attribute = face['faceAttributes']
                face_rect = face['faceRectangle']
                emotions = face['faceAttributes']['emotion']
                current_mood = max(emotions.items(), key=operator.itemgetter(1))[0]

                left, top, width, height = face_rect['left'], face_rect['top'], face_rect['width'], face_rect['height']
                (start_x, start_y), (end_x, end_y) = utils.detector_utils.get_rectangle(face)

                face_display = {
                    'gender': face_attribute['gender'],
                    'age': face_attribute['age'],
                    'mood': current_mood
                } 
            '''
    objects = ct.update(rectangles) 
    print("-----", face_display)

    for face in response:

        face_attribute = face['faceAttributes']
        face_rect = face['faceRectangle']
        emotions = face['faceAttributes']['emotion']
        current_mood = max(emotions.items(), key=operator.itemgetter(1))[0]

        left, top, width, height = face_rect['left'], face_rect['top'], face_rect['width'], face_rect['height']
        #(start_x, start_y), (end_x, end_y) = utils.detector_utils.get_rectangle(face)

        face_display = {
            'gender': face_attribute['gender'],
            'age': face_attribute['age'],
            'mood': current_mood
        } 

        for i, k  in enumerate(face_display):
            cv2.putText(frame, "{0}: {1}".format(k, face_display[k]),
                (left+width+5, top + 5 + 20*i),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                #(centroid[0] - 10*i, centroid[1]+20*i),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    for (object_ID, centroid) in objects.items():
        # Draw ID and centroid of the object in the output frame
        text = "ID {}".format(object_ID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        '''
        for i, k  in enumerate(face_display):
            print("Face Display: ", k, face_display[k])
            cv2.putText(frame, "{0}: {1}".format(k, face_display[k]),
                #(left+width+5, top + 5 + 20*i),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                (centroid[0] - 10*i, centroid[1]+20*i),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        '''

    info = [("Status: ", status)]
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    total_frames += 1
    fps.update()

fps.stop()
ct.total_detections()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Clean up
cap.stop()
cv2.destroyAllWindows()

