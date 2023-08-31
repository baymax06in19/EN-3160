import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

# Load the original image
img = cv.imread('images_01/spider.png', cv.IMREAD_COLOR).astype('uint8')

# Convert BGR image to RGB and then into  HSV color space
img_n = cv.cvtColor(img,cv.COLOR_BGR2RGB)
img_hsv = cv.cvtColor(img_n, cv.COLOR_RGB2HSV)
img_hsv_n = cv.cvtColor(img_hsv,cv.COLOR_BGR2RGB)

# Split the HSV image into H, S, and V channels
H, S, V = cv.split(img_hsv)


f , ax = plt.subplots(1,2)
ax[0].imshow(img_hsv_n)
ax[0].set_title('HSV_IMAGE')
ax[1].imshow(img_n)
ax[1].set_title('Original_IMG')
plt.show()

# Apply the saturation adjustment formula
a = 0.6  # Adjust this parameter as needed
sigma = 70  # Adjust this parameter as needed
S_adjustment = a * 128 * np.exp(-(S - 128) ** 2 / (2 * sigma ** 2)) ## s + s_adjustement can be negative also that is why we check the max of o, and the calculated value
S_corrected = np.minimum((S + S_adjustment), 255).astype('uint8')

# Merge the corrected S channel with the original H and V channels
img_hsv_corrected = cv.merge((H, S_corrected, V))

f , ax = plt.subplots(1,2)
ax[0].imshow(img_hsv_n)
ax[1].imshow(img_hsv_corrected)
plt.show()

# Convert the corrected HSV image to BGR color space for visualization
image_ve = cv.cvtColor(img_hsv_corrected, cv.COLOR_HSV2BGR)

# Convert the corrected HSV image to RGB color space
image_ve_n = cv.cvtColor(image_ve, cv.COLOR_BGR2RGB)

f , ax = plt.subplots(1,2)
ax[0].imshow(img_n)
ax[0].set_title('Original_IMG')
ax[1].imshow(image_ve_n)
ax[1].set_title('vibrance-enhanced image')

plt.show()

# Plot the relationship between S and S_corrected
plt.figure(figsize=(8, 6))
plt.scatter(S, S_corrected, color='blue', marker='o', alpha=0.5)
plt.xlabel('Original Saturation (S)')
plt.ylabel('Corrected Saturation (S_corrected)')
plt.title('S vs S_corrected')
plt.grid(True)
plt.show()



# Create a range of x values
x_values = np.linspace(0, 255, 256)

# Calculate the corresponding f(x) values using the formula
f_x_values = np.minimum(x_values + a * 128 * np.exp(-((x_values - 128) ** 2) / (2 * sigma ** 2)), 255)

# Plot the function
plt.plot(x_values, f_x_values, color='Green')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function f(x)')
plt.grid(True)
plt.show()


# Plot the original image, vibrance-enhanced image, and histograms
fig, axs = plt.subplots(2, 2, figsize=(7, 7))

# Display the original image
axs[0, 0].imshow(img_n)
axs[0, 0].set_title('Original Image')

# Display the vibrance-enhanced image
axs[0, 1].imshow(image_ve_n)
axs[0, 1].set_title('Vibrance-Enhanced Image')
#alpha=0.5: This parameter controls the transparency of the histogram bars. A value of 0.5 makes the bars somewhat transparent, allowing you to see the grid lines and any other elements in the background.
# Plot the original V channel histogram
axs[1, 0].hist(S.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.5) #.ravel() method is used to flatten the 2D array of the V channel into a 1D array, which is suitable for histogram plotting
axs[1, 0].set_title('Original S Channel Histogram')

# Plot the corrected V channel histogram
axs[1, 1].hist(S_corrected.ravel(), bins=256, range=[0, 256], color='red', alpha=0.5)
axs[1, 1].set_title('Corrected S Channel Histogram')

plt.tight_layout()
plt.show()

