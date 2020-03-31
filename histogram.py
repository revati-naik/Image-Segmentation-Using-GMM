import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import sys

def histogramPlot():
	# img = cv2.imread('./Data/buoy_frame.jpg')
	img = cv2.imread('./Data/train_set_orange/orange_buoy_21.jpg')
	img_test = cv2.imread('Data/buoy_frame.jpg')
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	mask = np.array(img_gray != 255, dtype=np.uint8)*255

	print("buoy frame:", img_test.shape)

	print("img:", img.shape)
	plt.figure("img red")
	plt.imshow(img[:,:,2])
	
	print("mask:", mask.shape)
	plt.figure("mask")
	plt.imshow(mask)
	plt.show()

	hist_1 = cv2.calcHist([img[:,:,2]], [0], mask, [256], [0,256])
	# hist_2 = cv2.calcHist([img], [1], mask, [256], [0,256])
	# hist_3 = cv2.calcHist([img], [2], mask, [256], [0,256])

	# color = ('b','g','r')
	# for i,col in enumerate(color):
	#     histr = cv2.calcHist([img[:,:,i]],[i],mask,[256],[0,256])
	#     plt.plot(histr,color = col)
	#     plt.xlim([0,256])
	# plt.show()

	plt.figure('R')
	plt.plot(hist_1)
	# plt.figure('G')
	# plt.plot(hist_2)
	# plt.figure('B')
	# plt.plot(hist_3)
	plt.show()
	sys.exit(0)

def testMain():
	histogramPlot()


if __name__ == '__main__':
	testMain()  