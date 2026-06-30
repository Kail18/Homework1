import cv2 as cv
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random
import shutil
from pathlib import Path


class DirectorySetup:
    """Central place for all input and output paths."""

    BASE_DIR = Path(__file__).resolve().parent

    IMAGE_PATH_OPTIONS = [
        BASE_DIR / "myImage.png",
        BASE_DIR / "python_script_og_image" / "myImage.png",
    ]

    OUTPUT_DIR = BASE_DIR / "generated_images"

    INPUT_DIR = OUTPUT_DIR / "input"
    COLOR_SPACE_DIR = OUTPUT_DIR / "color_spaces"
    AFFINE_DIR = OUTPUT_DIR / "affine_transformations"
    GAUSSIAN_DIR = OUTPUT_DIR / "gaussian_blur"
    PERSONAL_DIR = OUTPUT_DIR / "personal_experiment"

    LAPLACIAN_DIR = OUTPUT_DIR / "laplacian"
    SOBEL_DIR = OUTPUT_DIR / "sobel"
    CANNY_DIR = OUTPUT_DIR / "canny"
    PREWITT_DIR = OUTPUT_DIR / "prewitt"

    FIVE_IMAGE_PLOTS_DIR = OUTPUT_DIR / "five_image_plots"
    README_PLOTS_DIR = OUTPUT_DIR / "readme_plots"

    EDGE_METHOD_DIRS = {
        "laplacian": LAPLACIAN_DIR,
        "sobel": SOBEL_DIR,
        "canny": CANNY_DIR,
        "prewitt": PREWITT_DIR,
    }

    TRANSFORM_TYPES = [
        "input",
        "color_spaces",
        "affine_transformations",
        "gaussian_blur",
        "personal_experiment",
    ]

    def __init__(self):
        self.make_directories()

    def make_directories(self):
        base_folders = [
            self.OUTPUT_DIR,
            self.INPUT_DIR,
            self.COLOR_SPACE_DIR,
            self.AFFINE_DIR,
            self.GAUSSIAN_DIR,
            self.PERSONAL_DIR,
            self.FIVE_IMAGE_PLOTS_DIR,
            self.README_PLOTS_DIR,
        ]

        for folder in base_folders:
            folder.mkdir(parents=True, exist_ok=True)

        for edge_dir in self.EDGE_METHOD_DIRS.values():
            edge_dir.mkdir(parents=True, exist_ok=True)

            for transform_type in self.TRANSFORM_TYPES:
                (edge_dir / transform_type).mkdir(parents=True, exist_ok=True)

    def get_edge_output_path(self, method_name, transform_type, filename):
        return self.EDGE_METHOD_DIRS[method_name] / transform_type / filename

    def get_image_path(self):
        for path in self.IMAGE_PATH_OPTIONS:
            if path.exists():
                return path

        expected_paths = "\n".join(str(path) for path in self.IMAGE_PATH_OPTIONS)
        raise FileNotFoundError(
            "Could not find myImage.png. Put it in one of these locations:\n"
            f"{expected_paths}"
        )

class ImageWriter:
    """Small helper so every saved image is checked."""

    @staticmethod
    def save(path, image):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        success = cv.imwrite(str(path), image)
        if not success:
            raise IOError(f"Could not save image to: {path}")

        return path


class MyPlots:
    @staticmethod
    def safe_name(name):
        return name.replace(".", "_").replace(" ", "_").replace("/", "_")

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
    def make_five_image_plot(name, input_path, sample_num, output_path, setup, transform_type):
        paths = {
            "Sobel Edge": setup.get_edge_output_path(
                "sobel", transform_type, f"sobel_{name}.png"
            ),
            "Laplacian Edge": setup.get_edge_output_path(
                "laplacian", transform_type, f"laplacian_{name}.png"
            ),
            "Input Image": input_path,
            "Canny Edge": setup.get_edge_output_path(
                "canny", transform_type, f"canny_edges_{name}.png"
            ),
            "Prewitt Edge": setup.get_edge_output_path(
                "prewitt", transform_type, f"prewitt_combined_{name}.png"
            ),
        }

        images = {}
        for label, path in paths.items():
            if label == "Input Image":
                images[label] = cv.imread(str(path))
            else:
                images[label] = cv.imread(str(path), cv.IMREAD_GRAYSCALE)

            if images[label] is None:
                raise FileNotFoundError(f"Could not find image for {label}: {path}")

        fig = plt.figure(figsize=(12, 10))
        fig.patch.set_facecolor("#1e1e1e")

        fig.suptitle(
            MyPlots.pipeline_text(name, sample_num),
            color="cyan",
            fontsize=14,
            y=0.98,
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

        plt.savefig(
            str(output_path),
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()

    @staticmethod
    def create_42_plots_and_copy_6(subset, setup):
        all_plot_paths = []

        for i, (name, img, input_path, transform_type) in enumerate(subset):
            safe = MyPlots.safe_name(name)
            output_path = setup.FIVE_IMAGE_PLOTS_DIR / f"sample_{i}_{safe}.png"

            MyPlots.make_five_image_plot(
                name=name,
                input_path=input_path,
                sample_num=i,
                output_path=output_path,
                setup=setup,
                transform_type=transform_type,
            )

            all_plot_paths.append(output_path)

        chosen_plots = random.sample(all_plot_paths, 6)

        print("README plots:")
        for path in chosen_plots:
            copied_path = setup.README_PLOTS_DIR / path.name
            shutil.copy2(path, copied_path)
            print(copied_path)

        return chosen_plots


class ImageStats:
    def __init__(self, img):
        self.img = img
        self.pixel_data = img.flatten()

    def print_min_max_range(self):
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(self.img)
        range_val = max_val - min_val
        print(f"Min value: {min_val} at location {min_loc}")
        print(f"Max value: {max_val} at location {max_loc}")
        print(f"Range: {range_val}")

    def print_mean_std(self):
        mean_val, std_dev = cv.meanStdDev(self.img)
        print(f"Mean value: {mean_val[0][0]}")
        print(f"Standard deviation: {std_dev[0][0]}")

    def print_median_variance(self):
        median_val = np.median(self.pixel_data)
        variance_val = np.var(self.pixel_data)
        print(f"Median value: {median_val}")
        print(f"Variance: {variance_val}")

    def print_mode_value(self):
        mode_val, mode_count = stats.mode(self.pixel_data, keepdims=False)
        print(f"Mode value: {mode_val} with count {mode_count}")

    def print_skewness(self):
        skew_val = stats.skew(self.pixel_data)
        print(f"Skewness: {skew_val}")


class GaussianBlurApplication:
    def __init__(self, img_array, setup):
        self.img_array = img_array
        self.setup = setup

    def apply_gaussian_blur(self, kernel_size=15, sigmaX=None):
        if sigmaX is None:
            sigmaX = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

        new_img_array = []

        for name, img, image_path, transform_type in self.img_array:
            new_img_array.append((name, img, image_path, transform_type))

            for sigma in sigmaX:
                blurred_img = cv.GaussianBlur(img, (kernel_size, kernel_size), sigma)

                blur_name = f"{name}_blur_sigma_{sigma}"
                blur_path = (
                    self.setup.GAUSSIAN_DIR
                    / f"gaussian_blurred_{name}_k{kernel_size}_s{sigma}.png"
                )

                ImageWriter.save(blur_path, blurred_img)

                new_img_array.append(
                    (blur_name, blurred_img, blur_path, "gaussian_blur")
                )

        return new_img_array


class DetectionTechniques:
    def __init__(self, img_array, setup):
        self.img_array = img_array
        self.setup = setup

    @staticmethod
    def convert_to_gray(img):
        if len(img.shape) == 3:
            return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def normalize_to_uint8(img):
        normalized = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
        return np.uint8(normalized)

    def apply_sobel_detection(self):
        new_array = []

        for name, img, input_path, transform_type in self.img_array:
            gray_img = self.convert_to_gray(img)

            sobel_x = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize=5)
            sobel_y = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize=5)

            sobel_combined = cv.magnitude(sobel_x, sobel_y)
            sobel_display = self.normalize_to_uint8(sobel_combined)

            output_path = self.setup.get_edge_output_path(
                "sobel",
                transform_type,
                f"sobel_{name}.png",
            )

            ImageWriter.save(output_path, sobel_display)

            new_array.append(
                (f"sobel_{name}", sobel_display, output_path, transform_type)
            )

        return new_array

    def apply_laplacian_detection(self):
        new_array = []

        for name, img, input_path, transform_type in self.img_array:
            gray_img = self.convert_to_gray(img)

            blurred_gray = cv.GaussianBlur(gray_img, (3, 3), 0)

            laplacian = cv.Laplacian(blurred_gray, cv.CV_64F)
            laplacian_display = cv.convertScaleAbs(laplacian)

            output_path = self.setup.get_edge_output_path(
                "laplacian",
                transform_type,
                f"laplacian_{name}.png",
            )

            ImageWriter.save(output_path, laplacian_display)

            new_array.append(
                (f"laplacian_{name}", laplacian_display, output_path, transform_type)
            )

        return new_array

    def apply_canny_detection(self, threshold1=100, threshold2=200):
        new_array = []

        for name, img, input_path, transform_type in self.img_array:
            gray_img = self.convert_to_gray(img)

            canny_edges = cv.Canny(gray_img, threshold1, threshold2)

            output_path = self.setup.get_edge_output_path(
                "canny",
                transform_type,
                f"canny_edges_{name}.png",
            )

            ImageWriter.save(output_path, canny_edges)

            new_array.append(
                (f"canny_edges_{name}", canny_edges, output_path, transform_type)
            )

        return new_array

    def apply_prewitt_detection(self):
        new_array = []

        for name, img, input_path, transform_type in self.img_array:
            gray_img = self.convert_to_gray(img)

            kernelx = np.array(
                [[1, 0, -1],
                [1, 0, -1],
                [1, 0, -1]],
                dtype=np.float32,
            )

            kernely = np.array(
                [[1, 1, 1],
                [0, 0, 0],
                [-1, -1, -1]],
                dtype=np.float32,
            )

            prewitt_x = cv.filter2D(gray_img, cv.CV_32F, kernelx)
            prewitt_y = cv.filter2D(gray_img, cv.CV_32F, kernely)

            prewitt_combined = cv.magnitude(prewitt_x, prewitt_y)
            prewitt_display = self.normalize_to_uint8(prewitt_combined)

            output_path = self.setup.get_edge_output_path(
                "prewitt",
                transform_type,
                f"prewitt_combined_{name}.png",
            )

            ImageWriter.save(output_path, prewitt_display)

            new_array.append(
                (f"prewitt_combined_{name}", prewitt_display, output_path, transform_type)
            )

        return new_array


def main():
    setup = DirectorySetup()

    image_path = setup.get_image_path()
    img = cv.imread(str(image_path))

    if img is None:
        raise FileNotFoundError(f"OpenCV could not read image: {image_path}")

    original_input_path = setup.INPUT_DIR / "myImage.png"
    ImageWriter.save(original_input_path, img)

    grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Question 1: Basic image statistics of the grayscale version of the image.
    img_stats = ImageStats(grayscale_img)
    img_stats.print_min_max_range()
    img_stats.print_mean_std()
    img_stats.print_median_variance()
    img_stats.print_mode_value()
    img_stats.print_skewness()

    # Color-space conversions.
    grayscale_path = ImageWriter.save(setup.COLOR_SPACE_DIR / "grayscale_image.png", grayscale_img)

    _, binary_img = cv.threshold(grayscale_img, 127, 255, cv.THRESH_BINARY)
    binary_path = ImageWriter.save(setup.COLOR_SPACE_DIR / "binary_image.png", binary_img)

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv_path = ImageWriter.save(setup.COLOR_SPACE_DIR / "hsv_image.png", hsv_img)

    cielab_img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    cielab_path = ImageWriter.save(setup.COLOR_SPACE_DIR / "cielab_image.png", cielab_img)

    hls_img = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    hls_path = ImageWriter.save(setup.COLOR_SPACE_DIR / "hls_image.png", hls_img)

    # Lighting normalization by equalizing the HSV V channel.
    h, s, v = cv.split(hsv_img)
    v_equalized = cv.equalizeHist(v)
    hsv_equalized = cv.merge((h, s, v_equalized))
    result = cv.cvtColor(hsv_equalized, cv.COLOR_HSV2BGR)
    equalized_path = ImageWriter.save(setup.COLOR_SPACE_DIR / "equalized_image.png", result)

    # Original image affine transformations.
    img_warpAffine_left = cv.warpAffine(
        img,
        cv.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 45, 1),
        (img.shape[1], img.shape[0]),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    img_warpAffine_right = cv.warpAffine(
        img,
        cv.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), -45, 1),
        (img.shape[1], img.shape[0]),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    warped_left_path = ImageWriter.save(setup.AFFINE_DIR / "warped_image_left.png", img_warpAffine_left)
    warped_right_path = ImageWriter.save(setup.AFFINE_DIR / "warped_image_right.png", img_warpAffine_right)

    # Grayscale image affine transformations.
    grayscale_img_resize = cv.resize(
        grayscale_img,
        (grayscale_img.shape[1] // 2, grayscale_img.shape[0] // 2),
    )
    grayscale_img_reflection = cv.flip(grayscale_img, 1)

    grayscale_resize_path = ImageWriter.save(setup.AFFINE_DIR / "grayscale_image_resized.png", grayscale_img_resize)
    grayscale_reflection_path = ImageWriter.save(setup.AFFINE_DIR / "grayscale_image_reflection.png", grayscale_img_reflection)

    # Binary image affine transformations.
    binary_img_translation = cv.warpAffine(
        binary_img,
        np.float32([[1, 0, 50], [0, 1, 50]]),
        (binary_img.shape[1], binary_img.shape[0]),
    )
    binary_img_resize = cv.resize(
        binary_img,
        (binary_img.shape[1] // 2, binary_img.shape[0] // 2),
    )

    binary_translation_path = ImageWriter.save(setup.AFFINE_DIR / "binary_image_translated.png", binary_img_translation)
    binary_resize_path = ImageWriter.save(setup.AFFINE_DIR / "binary_image_resized.png", binary_img_resize)

    # HSV image affine transformations.
    hsv_img_shearing_one = cv.warpAffine(
        hsv_img,
        np.float32([[1, 0.5, 0], [0.5, 1, 0]]),
        (hsv_img.shape[1], hsv_img.shape[0]),
    )
    hsv_img_shearing_two = cv.warpAffine(
        hsv_img,
        np.float32([[1, -0.5, 0], [-0.5, 1, 0]]),
        (hsv_img.shape[1], hsv_img.shape[0]),
    )

    hsv_shear_one_path = ImageWriter.save(setup.AFFINE_DIR / "hsv_image_one_sheared.png", hsv_img_shearing_one)
    hsv_shear_two_path = ImageWriter.save(setup.AFFINE_DIR / "hsv_image_two_sheared.png", hsv_img_shearing_two)

    # CIELAB image affine transformations.
    cielab_img_rotation = cv.warpAffine(
        cielab_img,
        cv.getRotationMatrix2D((cielab_img.shape[1] / 2, cielab_img.shape[0] / 2), 30, 1),
        (cielab_img.shape[1], cielab_img.shape[0]),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    cielab_img_translation = cv.warpAffine(
        cielab_img,
        np.float32([[1, 0, -50], [0, 1, -50]]),
        (cielab_img.shape[1], cielab_img.shape[0]),
    )

    cielab_rotation_path = ImageWriter.save(setup.AFFINE_DIR / "cielab_image_rotated.png", cielab_img_rotation)
    cielab_translation_path = ImageWriter.save(setup.AFFINE_DIR / "cielab_image_translated.png", cielab_img_translation)

    # HLS image affine transformations.
    hls_img_rotation = cv.warpAffine(
        hls_img,
        cv.getRotationMatrix2D((hls_img.shape[1] / 2, hls_img.shape[0] / 2), -30, 1),
        (hls_img.shape[1], hls_img.shape[0]),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    hls_img_translation = cv.warpAffine(
        hls_img,
        np.float32([[1, 0, 50], [0, 1, -50]]),
        (hls_img.shape[1], hls_img.shape[0]),
    )

    hls_rotation_path = ImageWriter.save(setup.AFFINE_DIR / "hls_image_rotated.png", hls_img_rotation)
    hls_translation_path = ImageWriter.save(setup.AFFINE_DIR / "hls_image_translated.png", hls_img_translation)

    # Equalized image affine transformations.
    result_180_rotation = cv.warpAffine(
        result,
        cv.getRotationMatrix2D((result.shape[1] / 2, result.shape[0] / 2), 180, 1),
        (result.shape[1], result.shape[0]),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    result_shearing = cv.warpAffine(
        result,
        np.float32([[1, 0.5, 0.6], [0.5, 1, 0]]),
        (result.shape[1], result.shape[0]),
    )

    result_rotation_path = ImageWriter.save(setup.AFFINE_DIR / "equalized_image_rotated.png", result_180_rotation)
    result_shearing_path = ImageWriter.save(setup.AFFINE_DIR / "equalized_image_sheared.png", result_shearing)

    # Store each base image with its name and saved file path.
    image_array = [
        ("img", img, original_input_path, "input"),

        ("img_warpAffine_left", img_warpAffine_left, warped_left_path, "affine_transformations"),
        ("img_warpAffine_right", img_warpAffine_right, warped_right_path, "affine_transformations"),

        ("grayscale_img", grayscale_img, grayscale_path, "color_spaces"),
        ("grayscale_img_resize", grayscale_img_resize, grayscale_resize_path, "affine_transformations"),
        ("grayscale_img_reflection", grayscale_img_reflection, grayscale_reflection_path, "affine_transformations"),

        ("binary_img", binary_img, binary_path, "color_spaces"),
        ("binary_img_translation", binary_img_translation, binary_translation_path, "affine_transformations"),
        ("binary_img_resize", binary_img_resize, binary_resize_path, "affine_transformations"),

        ("hsv_img", hsv_img, hsv_path, "color_spaces"),
        ("hsv_img_shearing_one", hsv_img_shearing_one, hsv_shear_one_path, "affine_transformations"),
        ("hsv_img_shearing_two", hsv_img_shearing_two, hsv_shear_two_path, "affine_transformations"),

        ("cielab_img", cielab_img, cielab_path, "color_spaces"),
        ("cielab_img_rotation", cielab_img_rotation, cielab_rotation_path, "affine_transformations"),
        ("cielab_img_translation", cielab_img_translation, cielab_translation_path, "affine_transformations"),

        ("hls_img", hls_img, hls_path, "color_spaces"),
        ("hls_img_rotation", hls_img_rotation, hls_rotation_path, "affine_transformations"),
        ("hls_img_translation", hls_img_translation, hls_translation_path, "affine_transformations"),

        ("result", result, equalized_path, "color_spaces"),
        ("result_180_rotation", result_180_rotation, result_rotation_path, "affine_transformations"),
        ("result_shearing", result_shearing, result_shearing_path, "affine_transformations"),
    ]

    # Apply Gaussian blur to each base image.
    gaussian_blur = GaussianBlurApplication(image_array, setup)
    new_img_array = gaussian_blur.apply_gaussian_blur()
    print(f"Total original + Gaussian images: {len(new_img_array)}")

    personal_gaussian_blur_result = cv.GaussianBlur(result, (15, 15), 65)
    ImageWriter.save(setup.PERSONAL_DIR / "personal_gaussian_blur_result.png", personal_gaussian_blur_result)

    # Split into 4 subsets of 42 images.
    first_subset = new_img_array[:42]
    second_subset = new_img_array[42:84]
    third_subset = new_img_array[84:126]
    fourth_subset = new_img_array[126:168]

    print(f"First subset length: {len(first_subset)}")
    print(f"Second subset length: {len(second_subset)}")
    print(f"Third subset length: {len(third_subset)}")
    print(f"Fourth subset length: {len(fourth_subset)}")

    # Run edge detection on the fourth subset, which keeps your current homework flow.
    fourth_detection = DetectionTechniques(fourth_subset, setup)
    fourth_sobel = fourth_detection.apply_sobel_detection()
    fourth_laplacian = fourth_detection.apply_laplacian_detection()
    fourth_canny = fourth_detection.apply_canny_detection()
    fourth_prewitt = fourth_detection.apply_prewitt_detection()

    final_fourth_detection = (
        fourth_subset + fourth_sobel + fourth_laplacian + fourth_canny + fourth_prewitt
    )
    print(f"Final fourth detection length: {len(final_fourth_detection)}")

    # Create all 42 five-image plots and copy 6 random plots to the README folder.
    MyPlots.create_42_plots_and_copy_6(fourth_subset, setup)

    print(f"\nFinished. All generated files were saved under: {setup.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
