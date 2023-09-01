#Q8
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

original_images = ["images_01/im02small.png","images_01/im03small.png" ,"images_01/im09small.png" ,"images_01/im11small.png"]
zoom_outs = ["images_01/im02.png", "images_01/im03.png", "images_01/im09.png", "images_01/im11.png"]

def images_set():
 for j in range(4):
    image = cv.imread(original_images[j])
    image_zoom_out = cv.imread(zoom_outs[j])

    image_bilinear = cv.resize(image, None, fx=4, fy=4, interpolation=cv.INTER_LINEAR)
    image_near = cv.resize(image, None, fx=4, fy=4, interpolation=cv.INTER_NEAREST)
    
    # Create subplots
    fig, ax = plt.subplots(1, 4, figsize=(10, 5))  # Adjusted figsize for better layout

    # Plot the images
    ax[0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    ax[0].set_title("Original image")

    ax[1].imshow(cv.cvtColor(image_near, cv.COLOR_BGR2RGB))
    ax[1].set_title("Nearest-neighbor zoomed image")

    ax[2].imshow(cv.cvtColor(image_bilinear, cv.COLOR_BGR2RGB))
    ax[2].set_title("Bilinear interpolation zoomed image")

    ax[3].imshow(cv.cvtColor(image_zoom_out, cv.COLOR_BGR2RGB))
    ax[3].set_title("Zoomed-out version")

    # Customize the style
    plt.subplots_adjust(wspace=0.5)  # Adjust the horizontal space between subplots
    for axis in ax:
        axis.axis('off')  # Turn off axis labels and ticks
        axis.grid(False)  # Turn off grid lines
        axis.set_xticklabels([])  # Hide x-axis tick labels
        axis.set_yticklabels([])  # Hide y-axis tick labels
        axis.set_aspect('auto')  # Adjust aspect ratio if needed

    plt.suptitle("Image Zooming Examples", fontsize=16)  # Add a title for the entire subplot
    plt.tight_layout()  # Ensure tight layout
    plt.show()

images_set()
