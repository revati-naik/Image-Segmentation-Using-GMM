import matplotlib.pyplot as plt 
import numpy as np
import cv2 
import os


cropPoints = []
i = 0
def getPoint(event,x,y,flags,param):
	global cropPoints,i
	img = param[0]
	if event == cv2.EVENT_LBUTTONDOWN:
		mouseX, mouseY = x,y
		cropPoints.append([x,y])
		print("x, y:", mouseX, mouseY)

		# if len(cropPoints) == 5:
		# 	cropImage(cropPoints, img)
		
		if len(cropPoints) == 2:
			cropCircle(cropPoints, img,param[1])
			cropPoints=[]
			i+=1

def cropCircle(cropPoints,img,name):
	mask = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)

	circle_center = (int(0.5*(cropPoints[0][0] + cropPoints[1][0])),int(0.5*(cropPoints[0][1] + cropPoints[1][1])))
	radius = int(0.5*np.sqrt((cropPoints[0][0]-cropPoints[1][0])**2 + (cropPoints[0][1]-cropPoints[1][1])**2))
	# print(circle_center,radius)

	mask = cv2.circle(mask,circle_center,radius,255,-1)
	# print(mask.shape)
	# img = np.array(img,dtype=np.uint8)
	cropped = cv2.bitwise_and(img,img,mask=mask)
	cropped = cropped[circle_center[1]-30:circle_center[1]+30,circle_center[0]-30:circle_center[0]+30]
	# cv2.imshow("Cropped",cropped)
	name = name+str(i)+".jpg"
	print("Name: ",name)
	cv2.imwrite(name,cropped)
	# cv2.waitKey(0)






# def cropImage(cropPoints, img):
# 	# print("in crop image func")
# 	# print(type(cropPoints))
# 	new_list = []
# 	for i in cropPoints:
# 		new_list.append(np.array(i))
# 	new_list = np.array(new_list)
# 	mask = np.zeros(img.shape[0:2], dtype=np.uint8)

# 	cv2.drawContours(mask, [new_list], -1, (255,255,255), -1, cv2.LINE_AA)
# 	res = cv2.bitwise_and(img,img,mask=mask)
# 	rect = cv2.boundingRect(new_list)
# 	cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

# 	wbg = np.ones_like(img, np.uint8)*255
# 	cv2.bitwise_not(wbg, wbg, mask=mask)

# 	# Adding the two frames
# 	dst = wbg + res

# 	# Trying to display only the cropped image and removing the background
# 	x,y,c = np.where(dst != 255)
# 	top_left_x,top_left_y = np.min(x),np.min(y)
# 	bottom_right_x,bottom_right_y = np.max(x),np.max(y)
# 	cropped = dst[top_left_x-20:bottom_right_x+20,top_left_y-20:bottom_right_y+20]

# 	cv2.imshow('Cropped', cropped)
# 	cv2.imwrite('Data/train_set_orange/orange_buoy_20.jpg', cropped)
# 	cv2.waitKey(0)

# cv2.destroyAllWindows()

def testMain():
	img_path = '/home/default/ENPM673/Image-Segmentation-Using-GMM/Data/detectbuoy.avi'
	cap = cv2.VideoCapture(img_path)
	i = 0
	while True:
		save_name = "./Data/Proper_Dataset/orange_buoy/orange_"
		ret,img = cap.read()
		if ret==True:
			i+=1
		# print("Image size:",img.shape)
		if i%5==0:
			print("Frame :",i)	
			cv2.imshow('Image', img)
			cv2.setMouseCallback('Image', getPoint,[img,save_name])
			cv2.waitKey(0)

if __name__ == '__main__':
	testMain()