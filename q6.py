import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'images_01/jeniffer.jpg'  # Replace with the path to your image
img = cv.imread(image_path)

# Convert the image from BGR to HSV color space
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Split the HSV image into H, S, and V components
H, S, V = cv.split(hsv_img)


#mask the image
# Select the appropriate plane for thresholding to extract the foreground mask
#we use the Saturation (S) plane for thresholding
threshold_value = 13  # determines the point at which the pixels will be classified as either foreground or background.
foreground_mask = cv.threshold(S, threshold_value, 255, cv.THRESH_BINARY)[1]

# Obtain the foreground using cv2.bitwise_and
# Split the original image into color channels
B, G, R = cv.split(img)

# Extract the foreground for each color channel
foreground_B = cv.bitwise_and(B, B, mask=foreground_mask)
foreground_G = cv.bitwise_and(G, G, mask=foreground_mask)
foreground_R = cv.bitwise_and(R, R, mask=foreground_mask)

# Merge the foreground channels back into a single image
foreground = cv.merge([foreground_R, foreground_G, foreground_B])


# Calculate the histogram of the foreground
hist_foreground = cv.calcHist([foreground], [0], None, [256], [0, 256])

# Compute the cumulative sum of the histogram
cumulative_sum = np.cumsum(hist_foreground)
#print(cumulative_sum)

# Convert foreground image to grayscale if it's not already
if len(foreground.shape) > 2:
    foreground_gray = cv.cvtColor(foreground, cv.COLOR_BGR2GRAY)
else:
    foreground_gray = foreground

# Calculate histogram of the foreground
hist_foreground = cv.calcHist([foreground_gray], [0], None, [256], [0, 256])

# Compute the cumulative sum of the histogram
cumulative_sum = np.cumsum(hist_foreground)

# Apply histogram equalization to the foreground_gray image
equalized_foreground_gray = (cumulative_sum[foreground_gray] * 255.0 / cumulative_sum[-1]).astype(np.uint8)

# Create an empty 3-channel image for the equalized result
equalized_result = np.zeros_like(foreground)

# Assign equalized values to all three channels
equalized_result[:, :, 0] = equalized_foreground_gray
equalized_result[:, :, 1] = equalized_foreground_gray
equalized_result[:, :, 2] = equalized_foreground_gray



# Display the results
plt.figure(figsize=(12, 12))

plt.subplot(2, 3, 1)
plt.imshow(H, cmap='gray')
plt.title('Hue')

plt.subplot(2, 3, 2)
plt.imshow(S, cmap='gray')
plt.title('Saturation')

plt.subplot(2, 3, 3)
plt.imshow(V, cmap='gray')
plt.title('Value')

plt.subplot(2, 3, 4)
plt.imshow(foreground_mask, cmap='gray')
plt.title('Foreground Mask')

plt.subplot(2, 3, 5)
plt.imshow(foreground, cmap='gray')
plt.title('Foreground')

plt.subplot(2, 3, 6)
plt.imshow(equalized_result)
plt.title('Histogram-Equalized Result')

plt.tight_layout()
plt.show()

color = ('b', 'g', 'r')

plt.figure(figsize=(12, 6))

# Calculate histograms for the original foreground and the equalized result
plt.subplot(1, 2, 1)
for i, c in enumerate(color):
    hist_original = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist_original, color=c)
    
plt.xlim([0, 256])
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram Comparison - Original')


plt.subplot(1, 2, 2)
for i, c in enumerate(color):
    hist_equalized = cv.calcHist([equalized_result], [i], None, [256], [0, 256])
    plt.plot(hist_equalized, color=c)
    
plt.xlim([0, 256])
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram Comparison - Equalized')


plt.tight_layout()
plt.show()



background_mask_3d = 255 - foreground_mask
background_hsv = np.bitwise_and(hsv_img, background_mask_3d)   # Extract background
background_rgb = cv.cvtColor(background_hsv, cv.COLOR_HSV2RGB)
final_image = background_rgb + equalized_result     # Add with foreground

plt.figure(figsize = (10, 10))
plt.subplot(121)
plt.imshow(equalized_result)
plt.title('Equalized Foreground')
plt.axis('off')
plt.subplot(122)
plt.imshow(final_image)
plt.title('Final result with original background')
plt.axis('off')







