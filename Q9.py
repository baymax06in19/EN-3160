import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Load the input image
original_image = cv.imread("images_01/flower.jpg").astype('uint8')

# (a) Perform image segmentation
segmentation_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)

# Define a region of interest (ROI) within the image
roi_rect = (30, 30, original_image.shape[1] - 30, original_image.shape[0] - 150)
background_model = np.zeros((1, 65), dtype=np.float64)
foreground_model = np.zeros((1, 65), dtype=np.float64)
cv.grabCut(original_image, segmentation_mask, roi_rect, background_model, foreground_model, 5, cv.GC_INIT_WITH_RECT)

# Create a binary mask where 1 represents the foreground and 0 represents the background
binary_mask = np.where((segmentation_mask == 2) | (segmentation_mask == 0), 0, 1).astype('uint8')

# Apply the mask to the original image to extract the segmented foreground
segmented_foreground = original_image * binary_mask[:, :, np.newaxis]

# Create subplots for visualization
fig, ax = plt.subplots(1, 5, figsize=(15, 15),)

ax[0].imshow(cv.cvtColor(original_image, cv.COLOR_BGR2RGB))
ax[0].set_title('Original Image')


# Plot the segmentation mask
ax[1].imshow(binary_mask, cmap='gray')
ax[1].set_title('Segmentation Mask')

# Plot the segmented foreground
ax[2].imshow(cv.cvtColor(segmented_foreground, cv.COLOR_BGR2RGB))
ax[2].set_title('Segmented Foreground')

# Compute and plot the segmented background
segmented_background = original_image - segmented_foreground
ax[3].imshow(cv.cvtColor(segmented_background, cv.COLOR_BGR2RGB))
ax[3].set_title('Segmented Background')

# (b) Enhance the image
blurred_background = cv.GaussianBlur(original_image, (0, 0), 30)
enhanced_image = np.where(binary_mask[:, :, np.newaxis] == 1, original_image, blurred_background)


# Plot the enhanced image
ax[4].imshow(cv.cvtColor(enhanced_image, cv.COLOR_BGR2RGB))
ax[4].set_title('Enhanced Image')

plt.show()

