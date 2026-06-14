import cv2 as cv
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import random
import shutil

class MyPlots:
    @staticmethod
    def safe_name(name):
        return name.replace(".", "_").replace(" ", "_").replace("/", "_")

    @staticmethod
    def get_input_path(name, kernel_size=15):
        if "_blur_sigma_" in name:
            base_name, sigma = name.split("_blur_sigma_")
            return f"gaussian_blurred_{base_name}_k{kernel_size}_s{sigma}.png"

        original_name_map = {
            "img": "myImage.png",
            "img_warpAffine_left": "warped_image_left.png",
            "img_warpAffine_right": "warped_image_right.png",
            "grayscale_img": "grayscale_image.png",
            "grayscale_img_resize": "grayscale_image_resized.png",
            "grayscale_img_reflection": "grayscale_image_reflection.png",
            "binary_img": "binary_image.png",
            "binary_img__translation": "binary_image_translated.png",
            "binary_img_resize": "binary_image_resized.png",
            "hsv_img": "hsv_image.png",
            "hsv_img_shearing_one": "hsv_image_one_sheared.png",
            "hsv_img_shearing_two": "hsv_image_two_sheared.png",
            "cielab_img": "cielab_image.png",
            "cielab_img_rotation": "cielab_image_rotated.png",
            "cielab_img_translation": "cielab_image_translated.png",
            "hls_img": "hls_image.png",
            "hls_img_rotation": "hls_image_rotated.png",
            "hls_img_translation": "hls_image_translated.png",
            "result": "equalized_image.png",
            "result_180_rotation": "equalized_image_rotated.png",
            "result_shearing": "equalized_image_sheared.png",
        }

        return original_name_map[name]

    @staticmethod
    def pipeline_text(name, sample_num):
        text = f"Sample {sample_num} Pipeline Trajectory:\n"

        if "_blur_sigma_" in name:
            base_name, sigma = name.split("_blur_sigma_")
            text += f"{base_name}\n→ Gaussian Blur(k:15, σ:{sigma})"
        else:
            text += f"{name}\n→ No Gaussian Blur"

        return text

    @staticmethod
    def make_five_image_plot(name, sample_num, output_path, kernel_size=15):
        input_path = MyPlots.get_input_path(name, kernel_size)

        paths = {
            "Sobel Edge": f"sobel_{name}.png",
            "Laplacian Edge": f"laplacian{name}.png",
            "Input Image": input_path,
            "Canny Edge": f"canny_edges{name}.png",
            "Prewitt Edge": f"prewitt_combined{name}.png",
        }

        images = {}
        for label, path in paths.items():
            if label == "Input Image":
                images[label] = cv.imread(path)
            else:
                images[label] = cv.imread(path, cv.IMREAD_GRAYSCALE)

            if images[label] is None:
                raise FileNotFoundError(f"Could not find image for {label}: {path}")

        fig = plt.figure(figsize=(12, 10))
        fig.patch.set_facecolor("#1e1e1e")

        fig.suptitle(
            MyPlots.pipeline_text(name, sample_num),
            color="cyan",
            fontsize=14,
            y=0.98
        )

        positions = {
            "Sobel Edge": [0.35, 0.62, 0.30, 0.22],
            "Laplacian Edge": [0.02, 0.32, 0.30, 0.22],
            "Input Image": [0.35, 0.32, 0.30, 0.22],
            "Canny Edge": [0.68, 0.32, 0.30, 0.22],
            "Prewitt Edge": [0.35, 0.04, 0.30, 0.22],
        }

        for title, pos in positions.items():
            ax = fig.add_axes(pos)
            ax.set_title(title, color="white", fontsize=11)
            ax.axis("off")

            if title == "Input Image":
                ax.imshow(cv.cvtColor(images[title], cv.COLOR_BGR2RGB))
            else:
                ax.imshow(images[title], cmap="gray")

        plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=150)
        plt.close()

    @staticmethod
    def create_42_plots_and_copy_6(subset, output_folder="five_image_plots", readme_folder="readme_plots"):
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(readme_folder, exist_ok=True)

        all_plot_paths = []

        for i, (name, img) in enumerate(subset):
            safe = MyPlots.safe_name(name)
            output_path = os.path.join(output_folder, f"sample_{i}_{safe}.png")

            MyPlots.make_five_image_plot(
                name=name,
                sample_num=i,
                output_path=output_path
            )

            all_plot_paths.append(output_path)

        chosen_plots = random.sample(all_plot_paths, 6)

        print("README plots:")
        for path in chosen_plots:
            filename = os.path.basename(path)
            copied_path = os.path.join(readme_folder, filename)
            shutil.copy2(path, copied_path)
            print(copied_path)

        return chosen_plots

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
        for img, name in zip(self.imgArray, names):

            new_img_array.append((name,img))
            for sigma in sigmaX:
                blurred_img = cv.GaussianBlur(img, (kernel_size, kernel_size), sigma)
                blur_name = f"{name}_blur_sigma_{sigma}"
                new_img_array.append((blur_name, blurred_img))
                cv.imwrite(f'gaussian_blurred_{name}_k{kernel_size}_s{sigma}.png', blurred_img)
        
        return new_img_array
    
class DetectionTechniques:
    def __init__(self, imgArray):
        self.imgArray = imgArray
    
    """
        Method for doing a Sobel detection to a list of images

        Returns: a new list of images passed through Sobel detection
    """
    def apply_sobel_detection(self):
        new_array = []
        for name, img in self.imgArray:
            sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
            sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
            sobel_combined = cv.magnitude(sobel_x, sobel_y)
            cv.imwrite(f'sobel_{name}.png', sobel_combined)
            new_array.append(sobel_combined)
        return new_array
    
    """
        Method for doing a laplacian detection to a list of images

        Returns: a new list of images passed through laplacian detection
    """
    def apply_laplacian_detection(self):
        new_array = []
        for name, img in self.imgArray:
            laplacian = cv.Laplacian(img, cv.CV_64F)
            cv.imwrite(f'laplacian{name}.png', laplacian)
            new_array.append(laplacian)
        return new_array
    
    """
        Method for doing a canny detection to a list of images

        Returns: a new list of images passed through canny detection
    """
    def apply_canny_detection(self, threshold1=100, threshold2=200):
        new_array = []
        for name, img in self.imgArray:
            canny_edges = cv.Canny(img, threshold1, threshold2)
            cv.imwrite(f'canny_edges{name}.png', canny_edges)
            new_array.append(canny_edges)
        return new_array
    
    """
        Method for doing a prewitt detection to a list of images

        Returns: a new list of images passed through prewitt detection
    """

    def apply_prewitt_detection(self):
        new_array = []

        for name, img in self.imgArray:

            if len(img.shape) == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            kernelx = np.array([[1, 0, -1],
                                [1, 0, -1],
                                [1, 0, -1]], dtype=np.float32)

            kernely = np.array([[1, 1, 1],
                                [0, 0, 0],
                                [-1, -1, -1]], dtype=np.float32)

            prewitt_x = cv.filter2D(img, cv.CV_32F, kernelx)
            prewitt_y = cv.filter2D(img, cv.CV_32F, kernely)

            prewitt_combined = cv.magnitude(prewitt_x, prewitt_y)

            prewitt_display = cv.normalize(
                prewitt_combined,
                None,
                0,
                255,
                cv.NORM_MINMAX
            )

            prewitt_display = np.uint8(prewitt_display)

            cv.imwrite(f'prewitt_combined{name}.png', prewitt_display)

            new_array.append(prewitt_display)

        return new_array


def main() :
    img = cv.imread('myImage.png')
    grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Question 1: Find and print basic image statistics of the original image for each individual channel (min, max, average, median, mode, skew, range, standard deviation, variance)

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
    part2_length = len(new_img_array)
    print(part2_length)

    personal_gaussian_blur_result = cv.GaussianBlur(result, (15, 15), 65)
    cv.imwrite('personal_gaussian_blur_result.png', personal_gaussian_blur_result)

    # Split into 4 subsets of 42 images

    first_subset = new_img_array[:42]
    second_subset = new_img_array[42:84]
    third_subset = new_img_array[84:126]
    fourth_subset = new_img_array[126:168]

    len_first_subset = len(first_subset)
    print(len_first_subset)
    len_second_subset = len(second_subset)
    print(len_second_subset)
    len_third_subset = len(third_subset)
    print(len_third_subset)
    len_fourth_subset = len(fourth_subset)
    print(len_fourth_subset)

    #Fourth Detection

    fourth_detection = DetectionTechniques(fourth_subset)
    fourth_sobel = fourth_detection.apply_sobel_detection()
    fourth_laplacian = fourth_detection.apply_laplacian_detection()
    fourth_canny = fourth_detection.apply_canny_detection()
    fourth_prewitt = fourth_detection.apply_prewitt_detection()

    final_fourth_detection = (
        fourth_subset + fourth_sobel + fourth_laplacian + fourth_canny + fourth_prewitt
    )
    print(len(final_fourth_detection))

    # Plotting call

    MyPlots.create_42_plots_and_copy_6(fourth_subset)



        

    
if __name__ == "__main__":
    main()