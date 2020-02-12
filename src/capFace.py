import cv2

casc_path = 'data/haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(casc_path)

# grab the reference to the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# keep looping
while True:
	# grab the current frame
	ret, frame = cap.read()

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the videoy
	if frame is None:
		break

	# Detect objects in video stream, returns a list of rectangles
	faces = faceCascade.detectMultiScale(
      frame,
      scaleFactor=1.1,
      minNeighbors=5,
      minSize=(30, 30)
  	)

	# Draw rectangles around detected faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
		cv2.putText(frame, str(len(faces)),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,.7,(150,150,0),2)
		print("Found {0} faces!".format(len(faces)))

	# show the frame to our screen
	cv2.imshow("Video", frame)
	print(faces)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# close all windows
cap.release()
cv2.destroyAllWindows()
