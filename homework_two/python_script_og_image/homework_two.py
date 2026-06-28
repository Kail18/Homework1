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

class Clustering:
    def apply_kmeans_hsv(normalized_image, setup, k, target_cluster):
        hsv_image = cv.cvtColor(normalized_image, cv.COLOR_BGR2HSV)

        pixel_data = hsv_image.reshape((-1, 3)).astype(np.float32)

        criteria = (
            cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
            10,
            1.0
        )

        cv.setRNGSeed(42)

        compactness, labels, centers = cv.kmeans(
            pixel_data,
            k,
            None,
            criteria,
            10,
            cv.KMEANS_PP_CENTERS
        )

        labels_2d = labels.reshape(hsv_image.shape[:2])

        binary_mask = np.where(labels_2d == target_cluster, 255, 0).astype(np.uint8)

        segmented_image = cv.bitwise_and(
            normalized_image,
            normalized_image,
            mask=binary_mask
        )

        cv.imwrite(
            str(setup.KMEANS_DIR / f"kmeans_mask_k{k}_cluster{target_cluster}.png"),
            binary_mask
        )

        cv.imwrite(
            str(setup.KMEANS_DIR / f"kmeans_segmented_k{k}_cluster{target_cluster}.png"),
            segmented_image
        )

        return binary_mask, segmented_image, centers

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
    otsu_threshold, otsu_mask = cv.threshold(greyscale_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    adaptive_gaussian_mask = cv.adaptiveThreshold(greyscale_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    ostu_segmented = cv.bitwise_and(normalized_image, normalized_image, mask=otsu_mask)
    adaptive_segmented = cv.bitwise_and(normalized_image, normalized_image, mask=adaptive_gaussian_mask)

    cv.imwrite(str(setup.OTSU_DIR / "otsu_image.png"), otsu_mask)
    cv.imwrite(str(setup.OTSU_DIR / "otsu_segmented_image.png"), ostu_segmented)
    cv.imwrite(str(setup.ADAPTIVE_DIR / "adaptive_gaussian_image.png"), adaptive_gaussian_mask)
    cv.imwrite(str(setup.ADAPTIVE_DIR / "adaptive_segmented_image.png"), adaptive_segmented)

    ## K-means clustering in HSV color space

    myClustering = Clustering


    ## Testing 3 clusters
    # myClustering.apply_kmeans_hsv(normalized_image, setup, k=3, target_cluster=0)
    # myClustering.apply_kmeans_hsv(normalized_image, setup, k=3, target_cluster=1)
    # myClustering.apply_kmeans_hsv(normalized_image, setup, k=3, target_cluster=2)

    # ## Testing 4 clusters
    # myClustering.apply_kmeans_hsv(normalized_image, setup, k=4, target_cluster=0)
    # myClustering.apply_kmeans_hsv(normalized_image, setup, k=4, target_cluster=1)
    # myClustering.apply_kmeans_hsv(normalized_image, setup, k=4, target_cluster=2)
    # myClustering.apply_kmeans_hsv(normalized_image, setup, k=4, target_cluster=3)

    # ## Testing 5 clusters
    # myClustering.apply_kmeans_hsv(normalized_image, setup, k=5, target_cluster=0)
    # myClustering.apply_kmeans_hsv(normalized_image, setup, k=5, target_cluster=1)
    # myClustering.apply_kmeans_hsv(normalized_image, setup, k=5, target_cluster=2)
    # myClustering.apply_kmeans_hsv(normalized_image, setup, k=5, target_cluster=3)
    myClustering.apply_kmeans_hsv(normalized_image, setup, k=5, target_cluster=4)

   





if __name__ == "__main__":
    main()