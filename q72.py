import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
img = cv2.imread('images_01/einstein.png', cv2.IMREAD_GRAYSCALE)

# Define the Sobel horizontal and vertical kernels using the property
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

# Apply Sobel filtering using cv2.filter2D
img_sobel_x = cv2.filter2D(img, -1, sobel_x)
img_sobel_y = cv2.filter2D(img, -1, sobel_y)

# Calculate the magnitude of gradients
img_sobel = np.sqrt(img_sobel_x**2 + img_sobel_y**2)

# Display the original and Sobel-filtered images
plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(img_sobel_x, cmap='gray'), plt.title('Sobel X')
plt.subplot(133), plt.imshow(img_sobel, cmap='gray'), plt.title('Sobel Magnitude')
plt.show()