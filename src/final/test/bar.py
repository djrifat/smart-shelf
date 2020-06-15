
import imutils
import numpy as np
from utils.WebcamVideoCapture import WebcamVideoCapture
import os
import cv2,time


cap = WebcamVideoCapture(src=0).start()
time.sleep(2.0)

while True:

    frame = cap.read()
    frame2 = cap.read()
    cap.set_buffer(2)
    frame = imutils.resize(frame, width=500)
    frame2 = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hori = np.hstack((frame,frame2))
    #cv2.rectangle(frame, (0, 0), (50, 500), (255, 0, 0), -1)

    cv2.imshow("Mirabeau smart shelf", hori)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Stop FPS count and display the varbiables

cap.stop()
cv2.destroyAllWindows()


'''
#img = cv2.imread('test.png')
img = cv2.imread('roi.jpg')

img[100:150, 100:150] = [255,255,255] #(x,y, w,h)



#ball = img[280:340, 330:390]
#img[273:333, 100:160] = ball

#img = cv2.imread(filename)
#roi = img[row:row+height,column:column+width]
cv2.imshow("Foo", img)
print (type(img))
print ('Img shape: ', img.shape )       # Rows, cols, channels
print ('img.dtype: ', img.dtype)
print ('img.size: ', img.size)

cv2.waitKey(0)
cv2.destroyAllWindows()



'''
