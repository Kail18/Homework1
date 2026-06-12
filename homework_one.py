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

class GuassianBlurApplication:
    def __init__(self, imgArray):
        self.imgArray = imgArray

    """ This function applies a Gaussian blur to a list of images
        Parameters: kernel_size: The size of the kernel to be used for blurring (default is 3)
                    sigmaX: The standard deviation in the X direction for the Gaussian kernel (default is [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        Returns: returns a new_array of blurred images and original images
    
    """
    
    def apply_gaussian_blur(self, kernel_size = 15, sigmaX = None):
        if sigmaX is None:
            sigmaX = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        new_img_array = []
        names = [
            'img', 'img_warpAffine_left', 'img_warpAffine_right',
            'grayscale_img', 'grayscale_img_resize', 'grayscale_img_reflection',
            'binary_img', 'binary_img__translation', 'binary_img_resize',
            'hsv_img', 'hsv_img_shearing_one', 'hsv_img_shearing_two', 'cielab_img', 
            'cielab_img_rotation', 'cielab_img_translation', 'hls_img', 'hls_img_rotation', 
            'hls_img_translation', 'result', 'result_180_rotation', 'result_shearing'
            ]
        count = 0
        for img in self.imgArray:
            name = names[count]
            new_img_array.append(img)
            for sigma in sigmaX:
                blurred_img = cv.GaussianBlur(img, (kernel_size, kernel_size), sigma)
                new_img_array.append(blurred_img)
                cv.imwrite(f'gaussian_blurred_{name}_k{kernel_size}_s{sigma}.png', blurred_img)
            count += 1
        
        return new_img_array


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

    # Split the channels, equalize the V channel, and merge back to get the equalized image in HSV color space
    h, s, v = cv.split(hsv_img)

    v_equalized = cv.equalizeHist(v)
    hsv_equalized = cv.merge((h, s, v_equalized))
    result = cv.cvtColor(hsv_equalized, cv.COLOR_HSV2BGR)
    cv.imwrite('equalized_image.png', result)

    # Original Image Affine Transformations

    img_warpAffine_left = cv.warpAffine(img, cv.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 45, 1), (img.shape[1], img.shape[0]), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
    img_warpAffine_right = cv.warpAffine(img, cv.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), -45, 1), (img.shape[1], img.shape[0]), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
    cv.imwrite('warped_image_left.png', img_warpAffine_left)
    cv.imwrite('warped_image_right.png', img_warpAffine_right)

    # Grayscale Image Affine Transformations

    grayscale_img_resize = cv.resize(grayscale_img, (grayscale_img.shape[1] // 2, grayscale_img.shape[0] // 2))
    grayscale_img_reflection = cv.flip(grayscale_img, 1)
    cv.imwrite('grayscale_image_resized.png', grayscale_img_resize)
    cv.imwrite('grayscale_image_reflection.png', grayscale_img_reflection)

    #Binary Image Affine Transformations

    binary_img__translation = cv.warpAffine(binary_img, np.float32([[1, 0, 50], [0, 1, 50]]), (binary_img.shape[1], binary_img.shape[0]))
    binary_img_resize = cv.resize(binary_img, (binary_img.shape[1] // 2, binary_img.shape[0] // 2))
    cv.imwrite('binary_image_translated.png', binary_img__translation)
    cv.imwrite('binary_image_resized.png', binary_img_resize)

    #HSV Image Affine Transformations

    hsv_img_shearing_one = cv.warpAffine(hsv_img, np.float32([[1, 0.5, 0], [0.5, 1, 0]]), (hsv_img.shape[1], hsv_img.shape[0]))
    hsv_img_shearing_two = cv.warpAffine(hsv_img, np.float32([[1, -0.5, 0], [-0.5, 1, 0]]), (hsv_img.shape[1], hsv_img.shape[0]))
    cv.imwrite('hsv_image_one_sheared.png', hsv_img_shearing_one)
    cv.imwrite('hsv_image_two_sheared.png', hsv_img_shearing_two)

    #CIELAB Image Affine Transformations

    cielab_img_rotation = cv.warpAffine(cielab_img, cv.getRotationMatrix2D((cielab_img.shape[1] / 2, cielab_img.shape[0] / 2), 30, 1), (cielab_img.shape[1], cielab_img.shape[0]), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
    cielab_img_translation = cv.warpAffine(cielab_img, np.float32([[1, 0, -50], [0, 1, -50]]), (cielab_img.shape[1], cielab_img.shape[0]))
    cv.imwrite('cielab_image_rotated.png', cielab_img_rotation)
    cv.imwrite('cielab_image_translated.png', cielab_img_translation)

    #HLS Image Affine Transformations

    hls_img_rotation = cv.warpAffine(hls_img, cv.getRotationMatrix2D((hls_img.shape[1] / 2, hls_img.shape[0] / 2), -30, 1), (hls_img.shape[1], hls_img.shape[0]), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
    hls_img_translation = cv.warpAffine(hls_img, np.float32([[1, 0, 50], [0, 1, -50]]), (hls_img.shape[1], hls_img.shape[0]))
    cv.imwrite('hls_image_rotated.png', hls_img_rotation)
    cv.imwrite('hls_image_translated.png', hls_img_translation)

    # Equalized Image Affine Transformations
    result_180_rotation = cv.warpAffine(result, cv.getRotationMatrix2D((result.shape[1] / 2, result.shape[0] / 2), 180, 1), (result.shape[1], result.shape[0]), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
    result_shearing = cv.warpAffine(result, np.float32([[1, 0.5, 0.6], [0.5, 1, 0]]), (result.shape[1], result.shape[0]))
    cv.imwrite('equalized_image_rotated.png', result_180_rotation)
    cv.imwrite('equalized_image_sheared.png', result_shearing)

    # Create an array of the current Images

    image_array = [img, img_warpAffine_left, img_warpAffine_right, grayscale_img, grayscale_img_resize, grayscale_img_reflection, binary_img, binary_img__translation, binary_img_resize, hsv_img, hsv_img_shearing_one, hsv_img_shearing_two, cielab_img, cielab_img_rotation, cielab_img_translation, hls_img, hls_img_rotation, hls_img_translation, result, result_180_rotation, result_shearing]

    #Pass the images into the Gaussian Blur Class
    gaussian_blur = GuassianBlurApplication(image_array)
    new_img_array = gaussian_blur.apply_gaussian_blur()
    length = len(new_img_array)
    print(length)

    personal_gaussian_blur_result = cv.GaussianBlur(result, (15, 15), 65)
    cv.imwrite('personal_gaussian_blur_result.png', personal_gaussian_blur_result)




if __name__ == "__main__":
    main()