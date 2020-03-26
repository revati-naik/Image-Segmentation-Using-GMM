import matplotlib.pyplot as plt 
import numpy as np
import cv2 

cropPoints = []
def getPoint(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDOWN:
		mouseX, mouseY = x,y
		cropPoints.append((x,y))
		print("Points:", mouseX, mouseY)


def testMain():
	img = cv2.imread('Data/buoy_frame.jpg')
	print(img.shape)
	cv2.imshow('Image', img)
	cv2.setMouseCallback('Image', getPoint)
	cv2.waitKey(0)
	print(cropPoints)

if __name__ == '__main__':
	testMain()