import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image, num_bins=256):
    # Calculate histogram of the input image
    histogram, bins = np.histogram(image.flatten(), bins=num_bins, range=[0, 256])
    # Calculate cumulative distribution function (CDF)
    cdf = histogram.cumsum()
    # Normalize CDF to the dynamic range of the image
    cdf_normalized = cdf * (num_bins - 1) / cdf[-1]  
    # Map original intensities to new equalized intensities
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized) 
    # Reshape to the original image shape
    equalized_image = equalized_image.reshape(image.shape) 
    return equalized_image.astype(np.uint8)

# Load the image
img = cv.imread('images_01/shells.tif').astype('uint8')
img_equized = histogram_equalization(img)


f , ax = plt.subplots(2,2,figsize=(10,7))

ax[0,0].imshow(img)
ax[0,1].imshow(img_equized)

ax[1, 0].hist(img.ravel(), bins=256, range=[0, 256], color='red', alpha=0.5) 
ax[1, 0].set_title('img')

ax[1, 1].hist(img_equized.ravel(), bins=256, range=[0, 256], color='red', alpha=0.5) 
ax[1, 1].set_title('img_equized')

plt.show()