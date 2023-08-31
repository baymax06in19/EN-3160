import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Load the original image
img_org = cv.imread('images_01/highlights_and_shadows.jpg', cv.IMREAD_COLOR)

# Gamma correction factor
gamma = 0.51

# Convert BGR image to Lab color space
img_lab = cv.cvtColor(img_org, cv.COLOR_BGR2Lab)

# Split the Lab image into L, a, and b channels
L, a, b = cv.split(img_lab)

# Calculate the lookup table for gamma correction
table = np.array([(i / 255.0) ** (gamma) * 255.0 for i in np.arange(0, 256)]).astype('uint8')

# Apply gamma correction to the L channel using the lookup table
L_corrected = cv.LUT(L, table)
f ,ax = plt.subplots(1,2)
ax[0].imshow(img_lab)
ax[1].imshow(L_corrected)
plt.show()
# Merge the corrected L channel with the original a and b channels
img_lab_corrected = cv.merge((L_corrected, a, b))

# Convert the corrected Lab image back to RGB color space
img_gamma_corrected = cv.cvtColor(img_lab_corrected, cv.COLOR_Lab2BGR)

# Create subplots for displaying the original and gamma-corrected images
f, axarr = plt.subplots(2, 2)

# Display the original image
axarr[0,0].imshow(cv.cvtColor(img_org, cv.COLOR_BGR2RGB))
axarr[0,0].set_title('Original Image')

# Display the gamma-corrected image
axarr[0,1].imshow(cv.cvtColor(img_gamma_corrected, cv.COLOR_BGR2RGB))
axarr[0,1].set_title('Gamma Corrected Image')

color = ('b','g','r')
for i, c in enumerate(color):
    hist_orig = cv.calcHist([img_org], [i], None, [256], [0, 256])
    axarr[1, 0].plot(hist_orig, color=c)
    hist_gamma = cv.calcHist([img_gamma_corrected], [i], None, [256], [0, 256])
    axarr[1, 1].plot(hist_gamma, color=c)

# Display the subplots
plt.show()