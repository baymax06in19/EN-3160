import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

c = np.array([(50,50),(50,100),(150,255),(150,150)] ) # take cordinates (x,y) into an array

#make whole function as parts and then combined everything
t1=np.linspace(0,c[0,1],c[0,0]+1-0)
t2=np.linspace(c[0,1],c[1,1],c[1,0]-c[0,0])
t3=np.linspace(c[1,1],c[2,1],c[2,0]-c[1,0])
t4=np.linspace(c[2,1],c[3,1],c[3,0]-c[2,0])
t5=np.linspace(c[3,1],255,255-c[3,0])

transform=np.concatenate((t1,t2),axis=0).astype('uint8')
transform=np.concatenate((transform,t3),axis=0).astype('uint8')
transform=np.concatenate((transform,t4),axis=0).astype('uint8')
transform=np.concatenate((transform,t5),axis=0).astype('uint8')

fig,ax = plt.subplots()
ax.plot(transform)
ax.set_xlim(0,255) # used to set the x axis limited and fullfilled from 0 to 255
ax.set_ylim(0,255)
plt.show()


img_original = cv.imread('images_01/emma.jpg',cv.IMREAD_GRAYSCALE)
ax.set_aspect('equal')
plt.show()
cv.imshow("imageoriginal",img_original)
cv.waitKey(0)

image_transformed = cv.LUT(img_original,transform)

f,ax = plt.subplots(1,2)
ax[0].imshow(img_original,cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(image_transformed,cmap='gray')
ax[1].set_title('image_brighten')
plt.show()



