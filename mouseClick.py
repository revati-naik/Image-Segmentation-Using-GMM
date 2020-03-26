import matplotlib.pyplot as plt 
import numpy as np
import cv2 

cropPoints = []
def getPoint(event,x,y,flags,param):
	img = cv2.imread('Data/buoy_frame.jpg')
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
	for elem in cropPoints:
		new_list.append(np.array(elem))
	print(new_list)    
	new_list = np.array(new_list)
	mask = np.zeros(img.shape[0:2], dtype=np.uint8)

	cv2.drawContours(mask, [new_list], -1, (255,255,255), -1, cv2.LINE_AA)
	res = cv2.bitwise_and(img,img,mask=mask)
	rect = cv2.boundingRect(new_list)
	cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

	wbg = np.ones_like(img, np.uint8)*255
	cv2.bitwise_not(wbg, wbg, mask=mask)

	dst = wbg + res

	cv2.imshow('Cropped', dst)
	cv2.waitKey(0)


def testMain():
	img = cv2.imread('Data/buoy_frame.jpg')
	print("Image size:",img.shape)
	cv2.imshow('Image', img)
	cv2.setMouseCallback('Image', getPoint)
	cv2.waitKey(0)
	print(cropPoints)
	cropImage(cropPoints, img)

if __name__ == '__main__':
	testMain()