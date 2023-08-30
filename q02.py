import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

c = np.array([(150,0),(177,50),(183,170),(210,255)] ) # take cordinates (x,y) into an array

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

cn = np.array([(0,255),(60,255),(60,60),(255,255)] ) # take cordinates (x,y) into an array

#make whole function as parts and then combined everything
t1=np.linspace(0,cn[0,1],cn[0,0]+1-0)
t2=np.linspace(cn[0,1],cn[1,1],cn[1,0]-cn[0,0])
t3=np.linspace(cn[1,1],cn[2,1],cn[2,0]-cn[1,0])
t4=np.linspace(cn[2,1],cn[3,1],cn[3,0]-cn[2,0])
t5=np.linspace(cn[3,1],255,255-cn[3,0])

transform1=np.concatenate((t1,t2),axis=0).astype('uint8')
transform1=np.concatenate((transform1,t3),axis=0).astype('uint8')
transform1=np.concatenate((transform1,t4),axis=0).astype('uint8')
transform1=np.concatenate((transform1,t5),axis=0).astype('uint8')

cn = np.array([(0,255),(100,255),(100,0),(255,0)] ) # take cordinates (x,y) into an array

#make whole function as parts and then combined everything
t1=np.linspace(0,cn[0,1],cn[0,0]+1-0)
t2=np.linspace(cn[0,1],cn[1,1],cn[1,0]-cn[0,0])
t3=np.linspace(cn[1,1],cn[2,1],cn[2,0]-cn[1,0])
t4=np.linspace(cn[2,1],cn[3,1],cn[3,0]-cn[2,0])
t5=np.linspace(cn[3,1],255,255-cn[3,0])

transform2=np.concatenate((t1,t2),axis=0).astype('uint8')
transform2=np.concatenate((transform2,t3),axis=0).astype('uint8')
transform2=np.concatenate((transform2,t4),axis=0).astype('uint8')
transform2=np.concatenate((transform2,t5),axis=0).astype('uint8')


fig,ax = plt.subplots(1,3)
ax[0].plot(transform)
ax[0].set_xlim(0,255) # used to set the x axis limited and fullfilled from 0 to 255
ax[0].set_ylim(0,255)
ax[0].set_title("transform")
ax[1].plot(transform1)
ax[1].set_xlim(0,255) # used to set the x axis limited and fullfilled from 0 to 255
ax[1].set_ylim(0,255)
ax[1].set_title("transform1")
ax[2].plot(transform2)
ax[2].set_xlim(0,255) # used to set the x axis limited and fullfilled from 0 to 255
ax[2].set_ylim(0,255)
ax[2].set_title("transform2")
plt.show()


img_original = cv.imread('images_01/BrainProtonDensitySlice9.png',cv.IMREAD_GRAYSCALE)

cv.waitKey(0)

image_transformed = cv.LUT(img_original,transform)
image_transformed21 = cv.LUT(img_original,transform1)
image_transformed22 = cv.LUT(image_transformed21,transform)
image_transformed2 = cv.LUT(image_transformed22,transform2)

f,ax = plt.subplots(1,3)
ax[0].imshow(img_original,cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(image_transformed,cmap='gray')
ax[1].set_title('white matter')
ax[2].imshow(image_transformed2,cmap='gray')
ax[2].set_title('Gray matter')
plt.show()