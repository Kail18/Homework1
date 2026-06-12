import cv2 as cv
import numpy as np
from scipy import stats

class ImageStats:
    def __init__(self, img):
        self.img = img
        self.pixel_data = img.flatten()

    """
    print_min_max: Prints the minimum and maximum pixel values along with their locations.
        It also prints the range (difference between max and min) of the pixel values.
    """
    
    def print_min_max_range(self):
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(self.img)
        range_val = max_val - min_val
        print(f"Min value: {min_val} at location {min_loc}")
        print(f"Max value: {max_val} at location {max_loc}")
        print(f"Range: {range_val}")

    """
    print_mean_std: Prints the mean and standard deviation of the pixel values.
    """

    def print_mean_std(self):
        mean_val, std_dev = cv.meanStdDev(self.img)
        print(f"Mean value: {mean_val[0][0]}")
        print(f"Standard deviation: {std_dev[0][0]}")

    """
    print_median_variance: Prints the median and variance of the pixel values.
    """

    def print_median_variance(self):
        median_val = np.median(self.pixel_data)
        variance_val = np.var(self.pixel_data)
        print(f"Median value: {median_val}")
        print(f"Variance: {variance_val}")

    """
    print_mode_value: Prints the mode (most frequent pixel value) and its count.
    """

    def print_mode_value(self):
        mode_val, mode_count = stats.mode(self.pixel_data, keepdims=False)
        print(f"Mode value: {mode_val} with count {mode_count}")


    """
    print_skewness: Prints the skewness of the pixel value distribution, which indicates the asymmetry of the distribution.
    """

    def print_skewness(self):
        skew_val = stats.skew(self.pixel_data)
        print(f"Skewness: {skew_val}")
    
    


def main() :
    img = cv.imread('myimage.png')
    grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    """
    Question 1: nFind and print basic image statistics of the original image for each individual channel (min, max, average, median, mode, skew, range, standard deviation, variance)
    """

    img_stats = ImageStats(grayscale_img)
    img_stats.print_min_max_range()
    img_stats.print_mean_std()
    img_stats.print_median_variance()
    img_stats.print_mode_value()
    img_stats.print_skewness()

    #Gray scale image
    
    cv.imwrite('grayscale_image.png', grayscale_img)

    #Binary image

    _, binary_img = cv.threshold(grayscale_img, 127, 255, cv.THRESH_BINARY)
    cv.imwrite('binary_image.png', binary_img)

    #HSV image

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.imwrite('hsv_image.png', hsv_img)

    #CIELAB image

    cielab_img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    cv.imwrite('cielab_image.png', cielab_img)

    #HLS image

    hls_img = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    cv.imwrite('hls_image.png', hls_img)

    



if __name__ == "__main__":
    main()