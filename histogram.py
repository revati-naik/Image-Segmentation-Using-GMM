import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import sys

# def histogramPlot():
	# # img = cv2.imread('./Data/buoy_frame.jpg')
	# img = cv2.imread('./Data/train_set_orange/orange_buoy_21.jpg')
	# img_test = cv2.imread('Data/buoy_frame.jpg')
	# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# mask = np.array(img_gray != 255, dtype=np.uint8)*255

	# print("buoy frame:", img_test.shape)

	# print("img:", img.shape)
	# plt.figure("img red")
	# plt.imshow(img[:,:,2])
	
	# print("mask:", mask.shape)
	# plt.figure("mask")
	# plt.imshow(mask)
	# plt.show()

	# hist_1 = cv2.calcHist([img[:,:,2]], [0], mask, [256], [0,256])
	# # hist_2 = cv2.calcHist([img], [1], mask, [256], [0,256])
	# # hist_3 = cv2.calcHist([img], [2], mask, [256], [0,256])

	# # color = ('b','g','r')
	# # for i,col in enumerate(color):
	# #     histr = cv2.calcHist([img[:,:,i]],[i],mask,[256],[0,256])
	# #     plt.plot(histr,color = col)
	# #     plt.xlim([0,256])
	# # plt.show()

	# plt.figure('R')
	# plt.plot(hist_1)
	# # plt.figure('G')
	# # plt.plot(hist_2)
	# # plt.figure('B')
	# # plt.plot(hist_3)
	# plt.show()
	# sys.exit(0)
	


def testMain():
	# histogramPlot()
	img1 = cv2.imread('/home/default/ENPM673/Image-Segmentation-Using-GMM/Data/Proper_Dataset/green_buoy/green_0.jpg')
	fig , ax = plt.subplots(3,2)

	# histb = cv2.calcHist([img1],[0],None,[20],[0,256])
	# histg = cv2.calcHist([img1],[1],None,[20],[0,256])
	# histr = cv2.calcHist([img1],[2],None,[20],[0,256])

	# ax[0].plot(histb,'b') 
	# ax[1].plot(histg,'g') 
	# ax[2].plot(histr,'r') 
	# red = 
	ax[0,0].hist(img1[:,:,0].ravel(),range=(10,255),bins=256,color = 'b')
	ax[1,0].hist(img1[:,:,1].ravel(),range=(10,255),bins=256,color = 'g')
	ax[2,0].hist(img1[:,:,2].ravel(),range=(10,255),bins=256,color = 'r')
	
	ax[0,1].imshow(img1,cmap='gray')
	ax[1,1].imshow(img1,cmap='gray')
	ax[2,1].imshow(img1,cmap='gray')


	plt.show() 

if __name__ == '__main__':
	testMain()  