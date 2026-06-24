import cv2 as cv
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import random
import shutil
from pathlib import Path

from pathlib import Path
import cv2 as cv

class DirectorySetup:


    BASE_DIR = Path(__file__).resolve().parent
    IMAGE_PATH = BASE_DIR / "myImage.png"

    OUTPUT_DIR = BASE_DIR / "generated_images"

    CHANELS_DIR = OUTPUT_DIR / "channels"
    NORMALIZED_DIR = OUTPUT_DIR / "normalized"
    OTSU_DIR = OUTPUT_DIR / "otsu"
    ADAPTIVE_DIR = OUTPUT_DIR / "adaptive_gaussian"
    KMEANS_DIR = OUTPUT_DIR / "kmeans_hsv"
    MASKS_DIR = OUTPUT_DIR / "masks"
    PLOTS_DIR = OUTPUT_DIR / "plots"


    def make_directories(self):
        for folder in [
            self.CHANELS_DIR,
            self.NORMALIZED_DIR,
            self.OTSU_DIR,
            self.ADAPTIVE_DIR,
            self.KMEANS_DIR,
            self.MASKS_DIR,
            self.PLOTS_DIR,
        ]:
            folder.mkdir(parents=True, exist_ok=True)



def main():
    setup = DirectorySetup()
    setup.make_directories()

    img = cv.imread('homework_two/python_script_og_image/myImage.png')

    if img is None:
        print("Could not read the image.")
        return
    
    ## Split the image into its color channels
    
    blue_channel, green_channel, red_channel = cv.split(img)

    ## Equalize the histogram of each channel

    equalized_red_channel = cv.equalizeHist(red_channel)
    equalized_green_channel = cv.equalizeHist(green_channel)
    equalized_blue_channel = cv.equalizeHist(blue_channel)

    ## Merge the equalized channels back into a single image

    normalized_image = cv.merge((equalized_blue_channel, equalized_green_channel, equalized_red_channel))
    cv.imwrite(str(setup.NORMALIZED_DIR / "normalized_image.png"), normalized_image)

    ## Otsu and Adaptive Gaussian Thresholding
    greyscale_image = cv.cvtColor(normalized_image, cv.COLOR_BGR2GRAY)
    otsu_threshold, otsu_image = cv.threshold(greyscale_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    adaptive_gaussian_image = cv.adaptiveThreshold(greyscale_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    ostu_segmented = cv.bitwise_and(normalized_image, normalized_image, mask=otsu_image)
    adaptive_segmented = cv.bitwise_and(normalized_image, normalized_image, mask=adaptive_gaussian_image)

    cv.imwrite(str(setup.OTSU_DIR / "otsu_image.png"), otsu_image)
    cv.imwrite(str(setup.OTSU_DIR / "otsu_segmented_image.png"), ostu_segmented)
    cv.imwrite(str(setup.ADAPTIVE_DIR / "adaptive_gaussian_image.png"), adaptive_gaussian_image)
    cv.imwrite(str(setup.ADAPTIVE_DIR / "adaptive_segmented_image.png"), adaptive_segmented)








if __name__ == "__main__":
    main()