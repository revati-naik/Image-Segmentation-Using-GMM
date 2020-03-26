import matplotlib.pyplot as plt 
import numpy as np
import cv2 
import os

img_path = 'Data/frame_set/buoy_frame_95.jpg'

cropPoints = []
def getPoint(event,x,y,flags,param):
	img = cv2.imread(img_path)
	if event == cv2.EVENT_LBUTTONDOWN:
		mouseX, mouseY = x,y
		cropPoints.append((x,y))
		print("x, y:", mouseX, mouseY)

		if len(cropPoints) == 5:
			cropImage(cropPoints, img)

def cropImage(cropPoints, img):
	# print("in crop image func")
	# print(type(cropPoints))
	new_list = []
	for i in cropPoints:
		new_list.append(np.array(i))
	new_list = np.array(new_list)
	mask = np.zeros(img.shape[0:2], dtype=np.uint8)

	cv2.drawContours(mask, [new_list], -1, (255,255,255), -1, cv2.LINE_AA)
	res = cv2.bitwise_and(img,img,mask=mask)
	rect = cv2.boundingRect(new_list)
	cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

	wbg = np.ones_like(img, np.uint8)*255
	cv2.bitwise_not(wbg, wbg, mask=mask)

	# Adding the two frames
	dst = wbg + res

	# Trying to display only the cropped image and removing the background
	x,y,c = np.where(dst != 255)
	top_left_x,top_left_y = np.min(x),np.min(y)
	bottom_right_x,bottom_right_y = np.max(x),np.max(y)
	cropped = dst[top_left_x-20:bottom_right_x+20,top_left_y-20:bottom_right_y+20]

	cv2.imshow('Cropped', cropped)
	cv2.imwrite('Data/train_set_orange/orange_buoy_20.jpg', cropped)
	cv2.waitKey(0)

cv2.destroyAllWindows()

def testMain():
	img = cv2.imread(img_path)
	print("Image size:",img.shape)
	cv2.imshow('Image', img)
	cv2.setMouseCallback('Image', getPoint)
	cv2.waitKey(0)
	print(cropPoints)
	cropImage(cropPoints, img)

if __name__ == '__main__':
	testMain()