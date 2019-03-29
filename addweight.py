import os
import cv2
for root,dirs,files in os.walk('./CamVid/'):
	for file in files:
		if file.endswith(".png") and not file.endswith("_L.png"):
			a=cv2.imread(root+'/'+file)
			#print(root+file)
			#print(a)
			#cv2.imshow("win",a)
			a=cv2.resize(a,(640,640))
			head=file.split('.')[0]
			b=cv2.imread("./result/"+head+"_R.png")
			c=cv2.addWeighted(a,0.5,b,0.5,0)
			cv2.imwrite("./compare/"+head+"_C.png",c)
		

