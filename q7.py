import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def filter(image , kernel): 
    assert kernel.shape[0]%2 == 1 and kernel.shape[1]%2 == 1
    k_hh, k_hw = kernel.shape[0] // 2, kernel.shape[1] // 2
    h, w = image.shape
    image_float = cv.normalize(image.astype('float'), None, 0, 1, cv.NORM_MINMAX)
    result = np.zeros(image.shape, 'float')

    for m in range(k_hh, h - k_hh):
        for n in range(k_hw, w - k_hw):
            result[m, n] = np.dot(image_float[m-k_hh: m+k_hh+1, n-k_hw: n+k_hw+1].flatten(), kernel.flatten())

    result = result * 255   # Undo normalization
    result = np.minimum(255, np.maximum(0, result)).astype(np.uint8) # Limit between 0 and 255
    return result

# Define filtering for an already normalized image without any rounding
def filter_step(image, kernel):
    assert kernel.shape[0]%2 == 1 and kernel.shape[1]%2 == 1
    
    k_hh, k_hw = kernel.shape[0] // 2, kernel.shape[1] // 2
    h, w = image.shape
    result = np.zeros(image.shape, 'float')
    for m in range(k_hh, h - k_hh):
        for n in range(k_hw, w - k_hw):
            result[m, n] = np.dot(image[m-k_hh: m+k_hh+1, n-k_hw: n+k_hw+1].flatten(), kernel.flatten())
    return result

def filter_in_steps(image, kernel1, kernel2):
    
    image_float = cv.normalize(image.astype('float'), None, 0, 1, cv.NORM_MINMAX)
    result = filter_step(filter_step(image_float, kernel1), kernel2)
    result = result * 255
    result = np.minimum(255, np.maximum(0, result)).astype(np.uint8) # Limit between 0 and 255
    return result

img7 = cv.imread( "images_01/einstein.png", cv.IMREAD_GRAYSCALE)

# Sobel vertical kernel
kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
img7_a = cv.filter2D(img7, -1, kernel)  # Using filter2D
img7_b = filter(img7, kernel)   # Using custom function

kernel1 = np.array([1, 2, 1]).reshape((3, 1))
kernel2 = np.array([1, 0, -1]).reshape((1, 3))
img7_c = filter_in_steps(img7, kernel1, kernel2)

plt.figure(figsize = (15, 10))
plt.rc('axes', titlesize = 10)     # fontsize of the axes title
plt.subplot(141)
plt.imshow(img7, cmap = 'gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(142)
plt.imshow(img7_a, cmap = 'gray')
plt.title('Sobel Filter using filter2D')
plt.axis('off')
plt.subplot(143)
plt.imshow(img7_b, cmap = 'gray')
plt.title('Sobel Filter using custom function')
plt.axis('off')
plt.subplot(144)
plt.imshow(img7_c, cmap = 'gray')
plt.title('Sobel Filter using property of convolution')
plt.axis('off')
plt.tight_layout()
plt.show()