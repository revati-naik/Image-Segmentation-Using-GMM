import cv2
import numpy as np 

# Read video file
cap = cv2.VideoCapture('Data/detectbuoy.avi')

while True:
	# Capture frame by frame 
	ret, frame = cap.read()

	cv2.imshow('frame', frame)
	cv2.waitKey(2)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	cv2.imwrite('bouy_frame.jpg', frame)
	

cap.release()
cv2.destroyAllWindows()




