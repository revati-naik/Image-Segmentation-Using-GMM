import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os
from scipy.stats import norm


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
	


def histogramPlot():
	# histogramPlot()
	path = '/home/default/ENPM673/Image-Segmentation-Using-GMM/Data/Proper_Dataset/green_buoy/'
	images = os.listdir(path)
	# images = ['orange_1.jpg']

	for i in images:
		print(path+i)
		img1 = cv2.imread(path+i)
		fig , ax = plt.subplots(nrows=4,ncols=2)
		# plt.setp(ax, xticks=np.arange(50,255,50))
		# histb = cv2.calcHist([img1],[0],None,[20],[0,256])
		# histg = cv2.calcHist([img1],[1],None,[20],[0,256])
		# histr = cv2.calcHist([img1],[2],None,[20],[0,256])

		# ax[0].plot(histb,'b') 
		# ax[1].plot(histg,'g') 
		# ax[2].plot(histr,'r') 
		# red = 
		blue = (img1[:,:,0][img1[:,:,0]>50]).ravel()
		# histb,bin_edges = np.histogram(img1[:,:,0][img1[:,:,0]>50],bins = np.arange(0,256))
		# mu = np.mean(histb)
		# sigma = np.std(histb)
		# print(histb)
		# print(bin_edges)
		# ax[0,0].hist(blue,bins=256,color = 'b')
		# ax[0,0].plot(histb, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (histb - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
		mu = np.mean(blue)
		sigma = np.std(blue)
		count, bins, ignored = ax[0,0].hist(blue, 256, normed=True, color='b')
		ax[3,0].plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='b')
		ax[0,1].imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))

		# ax[1,0].hist((img1[:,:,1][img1[:,:,1]>50]).ravel(),bins=256,color = 'g')
		green = (img1[:,:,1][img1[:,:,1]>50]).ravel()
		mu = np.mean(green)
		sigma = np.std(green)
		count, bins, ignored = ax[1,0].hist(green, 256, normed=True, color='g')
		ax[3,0].plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='g')
		ax[1,1].imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
		
		# ax[2,0].hist((img1[:,:,2][img1[:,:,2]>50]).ravel(),bins=256,color = 'r')
		red = (img1[:,:,2][img1[:,:,2]>50]).ravel()
		mu = np.mean(red)
		sigma = np.std(red)
		count, bins, ignored = ax[2,0].hist(red, 256, normed=True, color='r')
		ax[3,0].plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
		ax[2,1].imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))

		plt.show() 

def checkThresh():
	cap = cv2.VideoCapture('/home/default/ENPM673/Image-Segmentation-Using-GMM/Data/detectbuoy.avi')

	while True:
		ret,img = cap.read()

		## Orange Buoy
		mask = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
		mask[np.where(img[:,:,2]>240)]=255
		# cv2.imshow("Mask :",mask)

		im2,contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

		if len(contours) != 0:
		    # draw in blue the contours that were founded
		    # cv2.drawContours(img, contours, -1, 255, 3)

		    # find the biggest countour (c) by the area
		    c = max(contours, key = cv2.contourArea)
		    x,y,w,h = cv2.boundingRect(c)

		    # draw the biggest contour (c) in green
		    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

		## Green Buoy
		mask = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
		mask[(img[:,:,1]>235) & (img[:,:,1]<245) & (img[:,:,2]>225) & (img[:,:,2]<240)]=255
		cv2.imshow("Mask G:",mask)

		im2,contours,hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

		if len(contours) != 0:
		    # draw in blue the contours that were founded
		    # cv2.drawContours(img, contours, -1, 255, 3)

		    # find the biggest countour (c) by the area
		    c = max(contours, key = cv2.contourArea)
		    x,y,w,h = cv2.boundingRect(c)

		    # draw the biggest contour (c) in green
		    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

		 
		cv2.imshow("Frame :",img)
		cv2.waitKey(0)


def EM():
	# histogramPlot()
	path = '/home/default/ENPM673/Image-Segmentation-Using-GMM/Data/Proper_Dataset/orange_buoy/'
	images = os.listdir(path)

	# for i in images:
	# 	img = cv2.imread(path+i)

	# 	hist = np.histogram(img,)
	# 	print(path+i)
	# 	cv2.imshow("Image",img)
	# 	cv2.waitKey(0)

	img = cv2.imread(path+'orange_'+str(1)+'.jpg')

	hist,bin_edges = np.histogram(img[:,:,2],bins = np.arange(0,256,dtype=np.uint8))
	print("hist :",hist)
	

def main():
	histogramPlot()
	# checkThresh()
	# EM()

if __name__ == '__main__':
	main()  