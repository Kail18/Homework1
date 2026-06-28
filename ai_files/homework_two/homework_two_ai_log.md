# All usage will be done with ChatGPT

## Prompt 1 13:34 6/24/26

1. Main project folder is Homework1 -> Venv, homework_one, homework_two, ai_files, and README.md. ai_files -> code_implementation.md, reasearch_ai.md. homework_one -> homework_one.py, myImage.png. homework_two -> homework_two.py, myImage.png

2. This was me giving information to the ai about how I had set up my project. In this response it also recommended that I setup the projects directories in a function. I instead did it using a class.

## Prompt 2 13:35 6/24/26

1. Also when I generate images I want to generate them into different folders based on the type of generation.

2. This was me setting some guidlines for the project

## Prompt 3 13:36 6/24/26

1. ok I have added img = cv.imread('myImage.png') to the main how do I split the image into seperate RGB channels

2. It gave me code to implement. First split the image into BGR variables then wrtie the channels.

## Prompt 4 13:54 6/24/26

1. The images are all just in black and white

2. According to ai the images are split into RGB but OpenCV saves it in black and white. It also provided a way to display the individual RGB colors but I decided against it since we only need to split images for the subsequent steps. I did not need the output directly.

## Prompt 5 14:10 6/24/26

1. Multi-Channel Color Normalization: Load the original image from Homework One. Split the image into its three color channels (e.g., R, G, and B or via the V channel in HSV / L channel in LAB). Apply Histogram Equalization independently to all three channels to normalize illumination and maximize contrast across the entire color spectrum. Merge the channels back together to create a fully normalized color image. Save this normalized color image; it will serve as the primary input for all subsequent segmentation tasks. What is the reasoning for these steps?

2. The main points were segmentation works best when the image has consistent lighting and stronger contrast.

## Prompt 6 14:41 6/24/26

1. how to do Otsu global threshold

2. First I need to convert the normalized image to grayscale. Then I implemented the recommended setup for calling the cv.threshold function.

## Prompt 7 15:29 6/24/26

1. how to do the color-space-clustering (kmeans)?

2. First convert the image to HSV. Then apply the K-means clustering. Then isolate the the

## Prompt 8 10:17 6/27/26

1. k_means = cv.kmeans(hsv_image.reshape((-1, 3)).astype(np.float32), 2, None, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv.KMEANS_RANDOM_CENTERS)[1] Break down all of the different parts to this line of code

2. It provided output that the line of code was too compressed which is subjective. .reshape converts the image from a 3D image to a 2D image. .astype is mapping the HSV pixel data to a float32. Convert the image into HSV pixels, group those pixels into 2 color clusters, then use the cluster labels to create a segmentation mask.

## Prompt 9 10:27 6/27/26

1. The k_means_mask was a black image is this a common result?

2. It stated that the mask was selected to be black, but can also be selected to be white if 255 was selected.

## Prompt 10 10:58 6/27/26

1. ## Three clusters

   k_means_three = cv.kmeans(hsv_image.reshape((-1, 3)).astype(np.float32), 3, None, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv.KMEANS_RANDOM_CENTERS)[1]

   k_means_mask_three = k_means_three.reshape(hsv_image.shape[:2]).astype(np.uint8)
   k_means_segmented_three = cv.bitwise_and(normalized_image, normalized_image, mask=k_means_mask_three)

   cv.imwrite(str(setup.KMEANS_DIR / "kmeans_mask_three.png"), k_means_mask_three)
   cv.imwrite(str(setup.KMEANS_DIR / "kmeans_segmented_image_three.png"), k_means_segmented_three)

   ## Four clusters

   k_means_four = cv.kmeans(hsv_image.reshape((-1, 3)).astype(np.float32), 4, None, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv.KMEANS_RANDOM_CENTERS)[1]

   k_means_mask_four = k_means_four.reshape(hsv_image.shape[:2]).astype(np.uint8)
   k_means_segmented_four = cv.bitwise_and(normalized_image, normalized_image, mask=k_means_mask_four)

   cv.imwrite(str(setup.KMEANS_DIR / "kmeans_mask_four.png"), k_means_mask_four)
   cv.imwrite(str(setup.KMEANS_DIR / "kmeans_segmented_image_four.png"), k_means_segmented_four)

   ## Five clusters

   k_means_five = cv.kmeans(hsv_image.reshape((-1, 3)).astype(np.float32), 5, None, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv.KMEANS_RANDOM_CENTERS)[1]

   k_means_mask_five = k_means_five.reshape(hsv_image.shape[:2]).astype(np.uint8)
   k_means_segmented_five = cv.bitwise_and(normalized_image, normalized_image, mask=k_means_mask_five)

   cv.imwrite(str(setup.KMEANS_DIR / "kmeans_mask_five.png"), k_means_mask_five)
   cv.imwrite(str(setup.KMEANS_DIR / "kmeans_segmented_image_five.png"), k_means_segmented_five)

   When I comment out my segment three and segment 5 my segment 4 becomes a different image. Why is that

   2. It told me my code was inefficient and I should have done a loop for each of the clusters. It also told me that I was using a random initialization so I should try PP_CENTERS for a stable center starting point.

   ## Prompt 11 11:03 6/27/26

   1. cv.KMEANS_PP_CENTERS What does this do?

   2. It told me what I already knew that it decreases the entropy of the clustering.

   ## Prompt 12 11:14 6/27/26

   1. Part 4: Classical & Optimization-Based Segmentation
      To handle the complex textures and shadows of the outdoor scene, move beyond simple pixel intensities. Use the fully normalized color image from Part 2 as the input for these methods:

   Color-Space Clustering (K-Means):
   Convert the normalized color image to the HSV color space.
   Apply K-Means clustering to segment the image into
   K
   distinct regions (test and select an optimal
   K
   value between 3 and 5).
   Isolate the cluster that most closely captures the "unknown figure."
   Save the resulting binary masks and the segmented foreground extractions for both methods.

   2. I went ahead and threw this in because I implemented the code output from the previous recommended code and it was not correct. This was to realign the parameters for the AI to give the type of code I was looking for.
