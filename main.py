import cv2
import numpy as np 

# Read video file
cap = cv2.VideoCapture('Data/detectbuoy.avi')
flag = 0

while True:
	# Capture frame by frame 
	ret, frame = cap.read()

	if ret == False:
		break

	if flag % 5 == 0:
		print("in if")
		cv2.imwrite("./Data/frame_set/buoy_frame_"+str(flag)+".jpg", frame)
		print("flag:", flag)
	
	flag += 1

cap.release()
cv2.destroyAllWindows()




