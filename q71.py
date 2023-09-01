import cv2
import matplotlib.pyplot as plt
import numpy as np

# Custom function for filtering
def filter(image, kernel):
    assert kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1
    k_hh, k_hw = kernel.shape[0] // 2, kernel.shape[1] // 2
    h, w = image.shape
    image_float = cv2.normalize(image.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
    result = np.zeros(image.shape, 'float')

    for m in range(k_hh, h - k_hh):
        for n in range(k_hw, w - k_hw):
            result[m, n] = np.dot(image_float[m - k_hh:m + k_hh + 1, n - k_hw:n + k_hw + 1].flatten(), kernel.flatten())

    result = result * 255  # Undo normalization
    result = np.minimum(255, np.maximum(0, result)).astype(np.uint8)  # Limit between 0 and 255
    return result

# Load the image
img = cv2.imread('images_01/einstein.png', cv2.IMREAD_GRAYSCALE)

# Define the Sobel horizontal and vertical kernels
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

# Apply Sobel filtering using the custom filter function
img_sobel_x = filter(img, sobel_x)
img_sobel_y = filter(img, sobel_y)

# Calculate the magnitude of gradients
img_sobel = np.sqrt(img_sobel_x**2 + img_sobel_y**2)

# Display the original and Sobel-filtered images
plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(img_sobel_x, cmap='gray'), plt.title('Sobel X')
plt.subplot(133), plt.imshow(img_sobel, cmap='gray'), plt.title('Sobel Magnitude')
plt.show()
