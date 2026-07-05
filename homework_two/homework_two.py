import cv2 as cv
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import random
import shutil
from pathlib import Path

## For Homework Three I will seperate these into diffent files and I will do the same for homework 1


class DirectorySetup:
    BASE_DIR = Path(__file__).resolve().parent
    IMAGE_PATH = BASE_DIR / "myImage.png"

    OUTPUT_DIR = BASE_DIR / "generated_images"

    NORMALIZED_DIR = OUTPUT_DIR / "normalized"
    OTSU_DIR = OUTPUT_DIR / "otsu"
    ADAPTIVE_DIR = OUTPUT_DIR / "adaptive_gaussian"
    KMEANS_DIR = OUTPUT_DIR / "kmeans_hsv"
    MASKS_DIR = OUTPUT_DIR / "masks"
    PLOTS_DIR = OUTPUT_DIR / "plots"
    GROUND_TRUTH_DIR = OUTPUT_DIR / "ground_truth"

    def make_directories(self):
        for folder in [
            self.NORMALIZED_DIR,
            self.OTSU_DIR,
            self.ADAPTIVE_DIR,
            self.KMEANS_DIR,
            self.MASKS_DIR,
            self.PLOTS_DIR,
            self.GROUND_TRUTH_DIR
        ]:
            folder.mkdir(parents=True, exist_ok=True)

class MiscMethods:

    """ This function saves a binary mask and segmeted image for a given cluster in the HSV color space using K-means clustering.
        Args: normalized_image: The input image that has been normalized.
                setup: An instance of the DirectorySetup class to manage output directories.
                k: The number of clusters to form.
                target_cluster: The specific cluster to extract and save as a binary mask and segmented image.
        Returns:  binary_mask: A binary mask where the pixels belonging to the target cluster are white (255) and all others are black (0).
                    segmented_image: The original image with only the pixels of the target cluster retained, and all other pixels set to black.
                    centers: The centers of the clusters formed by K-means clustering in the HSV color space.
    """

    def apply_kmeans_hsv(self, normalized_image, setup, k, target_cluster):
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

        return binary_mask, segmented_image
    

    """ This function uses GrabCut and a manually decided rectangle to create a rough draft ground truth"""
    def create_grabcut_ground_truth(self, img, setup):

        # Format: x, y, width, height
        rect = (1261, 169, 905, 1318)

        mask = np.zeros(img.shape[:2], np.uint8)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv.grabCut(
            img,
            mask,
            rect,
            bgd_model,
            fgd_model,
            5,
            cv.GC_INIT_WITH_RECT
        )

        # Convert GrabCut labels into a binary mask
        ground_truth_mask = np.where(
            (mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD),
            255,
            0
        ).astype("uint8")

        ground_truth_segmented = cv.bitwise_and(
            img,
            img,
            mask=ground_truth_mask
        )

        cv.imwrite(str(setup.GROUND_TRUTH_DIR / "ground_truth_mask_initial.png"), ground_truth_mask)
        cv.imwrite(str(setup.GROUND_TRUTH_DIR / "ground_truth_segmented_initial.png"), ground_truth_segmented)

        return ground_truth_mask
    
    """This function allows the user to manually draw a polygon around the unknown figure in the image to create a ground truth mask and segmented image."""
    def create_polygon_ground_truth(self, img, setup):

        points = []
        display_img = img.copy()

        def click_event(event, x, y, flags, param):
            nonlocal points, display_img

            if event == cv.EVENT_LBUTTONDOWN:
                points.append((x, y))

                # Draw point
                cv.circle(display_img, (x, y), 4, (0, 0, 255), -1)

                # Draw line from previous point to current point
                if len(points) > 1:
                    cv.line(display_img, points[-2], points[-1], (0, 0, 255), 2)

                cv.imshow("Draw Ground Truth Polygon", display_img)

        cv.namedWindow("Draw Ground Truth Polygon")
        cv.setMouseCallback("Draw Ground Truth Polygon", click_event)

        print("Instructions:")
        print("Left click around the unknown figure.")
        print("Press u to undo the last point.")
        print("Press c to close the polygon preview.")
        print("Press s to save the mask.")
        print("Press q to quit without saving.")

        while True:
            cv.imshow("Draw Ground Truth Polygon", display_img)
            key = cv.waitKey(1) & 0xFF

            if key == ord("u"):
                if points:
                    points.pop()
                    display_img = img.copy()

                    for i, point in enumerate(points):
                        cv.circle(display_img, point, 4, (0, 0, 255), -1)

                        if i > 0:
                            cv.line(display_img, points[i - 1], points[i], (0, 0, 255), 2)

                    cv.imshow("Draw Ground Truth Polygon", display_img)

            elif key == ord("c"):
                if len(points) > 2:
                    cv.line(display_img, points[-1], points[0], (0, 0, 255), 2)
                    cv.imshow("Draw Ground Truth Polygon", display_img)

            elif key == ord("s"):
                if len(points) < 3:
                    print("You need at least 3 points to create a polygon.")
                    continue

                mask = np.zeros(img.shape[:2], dtype=np.uint8)

                polygon = np.array(points, dtype=np.int32)
                cv.fillPoly(mask, [polygon], 255)

                segmented = cv.bitwise_and(img, img, mask=mask)

                cv.imwrite(
                    str(setup.GROUND_TRUTH_DIR / "ground_truth_mask_final.png"),
                    mask
                )

                cv.imwrite(
                    str(setup.GROUND_TRUTH_DIR / "ground_truth_segmented_final.png"),
                    segmented
                )

                return mask, segmented

                print("Saved ground_truth_mask_final.png")
                print("Saved ground_truth_segmented_final.png")
                break

            elif key == ord("q"):
                print("Quit without saving.")
                break

        cv.destroyAllWindows()

    def make_binary_mask(self, mask):
        """
        Converts a grayscale mask into a binary mask with values 0 and 1.
        """
        _, binary_mask = cv.threshold(mask, 127, 1, cv.THRESH_BINARY)
        return binary_mask.astype(bool)


    def calculate_iou(self, ground_truth_mask, predicted_mask):
        """
        Calculates Intersection over Union, also called Jaccard Index.
        """
        ground_truth_binary = self.make_binary_mask(ground_truth_mask)
        predicted_binary = self.make_binary_mask(predicted_mask)

        intersection = np.logical_and(ground_truth_binary, predicted_binary)
        union = np.logical_or(ground_truth_binary, predicted_binary)

        if np.sum(union) == 0:
            return 0.0

        iou = np.sum(intersection) / np.sum(union)
        return iou


    def calculate_dice(self, ground_truth_mask, predicted_mask):
        """
        Calculates Dice Similarity Coefficient.
        """
        ground_truth_binary = self.make_binary_mask(ground_truth_mask)
        predicted_binary = self.make_binary_mask(predicted_mask)

        intersection = np.logical_and(ground_truth_binary, predicted_binary)

        total_pixels = np.sum(ground_truth_binary) + np.sum(predicted_binary)

        if total_pixels == 0:
            return 0.0

        dice = (2 * np.sum(intersection)) / total_pixels
        return dice
    
    def evaluate_segmentation_methods(self, setup):
        ground_truth_mask = cv.imread(
            str(setup.MASKS_DIR / "binary_ground_truth_mask.png"),
            cv.IMREAD_GRAYSCALE
        )

        otsu_mask = cv.imread(
            str(setup.MASKS_DIR / "binary_otsu_mask.png"),
            cv.IMREAD_GRAYSCALE
        )

        adaptive_mask = cv.imread(
            str(setup.MASKS_DIR / "binary_adaptive_mask.png"),
            cv.IMREAD_GRAYSCALE
        )

        kmeans_mask = cv.imread(
            str(setup.MASKS_DIR / "binary_kmeans_mask.png"),
            cv.IMREAD_GRAYSCALE
        )

        if ground_truth_mask is None:
            print("Could not load binary ground truth mask.")
            return

        masks = {
            "Otsu Thresholding": otsu_mask,
            "Adaptive Gaussian Thresholding": adaptive_mask,
            "K-Means HSV Clustering": kmeans_mask
        }

        print("\nQuantitative Segmentation Evaluation")
        print("-----------------------------------")

        for method_name, predicted_mask in masks.items():
            if predicted_mask is None:
                print(f"{method_name}: mask not found.")
                continue

            if predicted_mask.shape != ground_truth_mask.shape:
                predicted_mask = cv.resize(
                    predicted_mask,
                    (ground_truth_mask.shape[1], ground_truth_mask.shape[0]),
                    interpolation=cv.INTER_NEAREST
                )

            iou = self.calculate_iou(ground_truth_mask, predicted_mask)
            dice = self.calculate_dice(ground_truth_mask, predicted_mask)

            print(f"{method_name}")
            print(f"  IoU / Jaccard Index: {iou:.4f}")
            print(f"  Dice Coefficient:   {dice:.4f}")
            print()

    def create_segmentation_comparison_plot(self, img, normalized_image, setup):
        otsu_mask = cv.imread(
            str(setup.MASKS_DIR / "binary_otsu_mask.png"),
            cv.IMREAD_GRAYSCALE
        )

        adaptive_mask = cv.imread(
            str(setup.MASKS_DIR / "binary_adaptive_mask.png"),
            cv.IMREAD_GRAYSCALE
        )

        kmeans_mask = cv.imread(
            str(setup.MASKS_DIR / "binary_kmeans_mask.png"),
            cv.IMREAD_GRAYSCALE
        )

        ground_truth_mask = cv.imread(
            str(setup.MASKS_DIR / "binary_ground_truth_mask.png"),
            cv.IMREAD_GRAYSCALE
        )

        if otsu_mask is None or adaptive_mask is None or kmeans_mask is None or ground_truth_mask is None:
            print("One or more masks could not be loaded for the comparison plot.")
            return

        # Convert OpenCV BGR images to RGB for matplotlib
        original_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        normalized_rgb = cv.cvtColor(normalized_image, cv.COLOR_BGR2RGB)

        images = [
            original_rgb,
            normalized_rgb,
            otsu_mask,
            adaptive_mask,
            kmeans_mask,
            ground_truth_mask
        ]

        titles = [
            "Original Image",
            "Normalized Color Image",
            "Otsu Mask",
            "Adaptive Gaussian Mask",
            "K-Means Mask",
            "Ground Truth Mask"
        ]

        plt.figure(figsize=(18, 6))

        for i, image in enumerate(images):
            plt.subplot(1, 6, i + 1)

            if len(image.shape) == 2:
                plt.imshow(image, cmap="gray")
            else:
                plt.imshow(image)

            plt.title(titles[i], fontsize=10)
            plt.axis("off")

        plt.tight_layout()

        output_path = setup.PLOTS_DIR / "segmentation_comparison.png"
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved comparison plot to: {output_path}")

def main():
    setup = DirectorySetup()
    setup.make_directories()

    img = cv.imread(str(setup.IMAGE_PATH))

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

    cv.imwrite(str(setup.OTSU_DIR / "otsu_mask.png"), otsu_mask)
    cv.imwrite(str(setup.OTSU_DIR / "otsu_segmented_image.png"), ostu_segmented)
    cv.imwrite(str(setup.ADAPTIVE_DIR / "adaptive_gaussian_image.png"), adaptive_gaussian_mask)
    cv.imwrite(str(setup.ADAPTIVE_DIR / "adaptive_segmented_image.png"), adaptive_segmented)

    ## K-means clustering in HSV color space

    methods = MiscMethods()


    ## Testing 3 clusters
    # methods.apply_kmeans_hsv(normalized_image, setup, k=3, target_cluster=0)
    # methods.apply_kmeans_hsv(normalized_image, setup, k=3, target_cluster=1)
    # methods.apply_kmeans_hsv(normalized_image, setup, k=3, target_cluster=2)

    # ## Testing 4 clusters
    # methods.apply_kmeans_hsv(normalized_image, setup, k=4, target_cluster=0)
    # methods.apply_kmeans_hsv(normalized_image, setup, k=4, target_cluster=1)
    # methods.apply_kmeans_hsv(normalized_image, setup, k=4, target_cluster=2)
    # methods.apply_kmeans_hsv(normalized_image, setup, k=4, target_cluster=3)

    # ## Testing 5 clusters
    # methods.apply_kmeans_hsv(normalized_image, setup, k=5, target_cluster=0)
    # methods.apply_kmeans_hsv(normalized_image, setup, k=5, target_cluster=1)
    # methods.apply_kmeans_hsv(normalized_image, setup, k=5, target_cluster=2)
    # methods.apply_kmeans_hsv(normalized_image, setup, k=5, target_cluster=3)
    k_mean_mask, k_mean_segmented = methods.apply_kmeans_hsv(normalized_image, setup, k=5, target_cluster=4)


    # This was an attempt at using GrabCut which did not work well enough to use.
    # methods.create_grabcut_ground_truth(img, setup)

    ## This is to determine the rectangle dimensions for GrabCut.
    # rect = cv.selectROI("Select unknown figure", img, fromCenter=False, showCrosshair=True)
    # cv.destroyWindow("Select unknown figure")

    # print("Selected rectangle:", rect)

    # UNCOMMENT out the line of code below this if you need to make the ground truth
    # ground_truth_mask, ground_truth_segmented = methods.create_polygon_ground_truth(img, setup)

    
    ## Converting all the masks to binary masks for evaluation

    binary_otsu_mask = methods.make_binary_mask(otsu_mask)
    if binary_otsu_mask is None:
        print("Could not create binary mask for Otsu.")
        return

    binary_adaptive_mask = methods.make_binary_mask(adaptive_gaussian_mask)
    binary_kmeans_mask = methods.make_binary_mask(k_mean_mask)


    ground_truth_mask = cv.imread(
        str(setup.GROUND_TRUTH_DIR / "ground_truth_mask_final.png"),
        cv.IMREAD_GRAYSCALE
    )

    if ground_truth_mask is None:
        print("Could not load ground truth mask.")
        return

    binary_ground_truth = methods.make_binary_mask(ground_truth_mask)

    cv.imwrite(str(setup.MASKS_DIR / "binary_otsu_mask.png"), binary_otsu_mask.astype(np.uint8) * 255)
    cv.imwrite(str(setup.MASKS_DIR / "binary_adaptive_mask.png"), binary_adaptive_mask.astype(np.uint8) * 255)
    cv.imwrite(str(setup.MASKS_DIR / "binary_kmeans_mask.png"), binary_kmeans_mask.astype(np.uint8) * 255)
    cv.imwrite(str(setup.MASKS_DIR / "binary_ground_truth_mask.png"), binary_ground_truth.astype(np.uint8) * 255)

   
    ## Evaluate segmentation methods
    methods.evaluate_segmentation_methods(setup)

    ## Generate the Plot for the README

    methods.create_segmentation_comparison_plot(img, normalized_image, setup)




if __name__ == "__main__":
    main()