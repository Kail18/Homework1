# Homework 2

## Qualitative Analysis

### Adaptive Gaussian Thresholding

1. Pro: Adaptive does better than otsu at handling images with uneven lighting. Typically this is used for outdoor images since the lighting is uneven with different background textures

2. Con: Secondary to a cluttered small background detail or a poor block size the adaptive threshold can become noisy and include too much background noise.

3. Background Noise: Since the algorithm focuses on local pixel areas the background with highly detailed textures may get pulled to the foreground. This occurs since the method might treat these texture changes as important.

4. Color Normalization: This improves the contrast locally which can help the algorithm detect the unknown figure more clearly. Normalized images can also provide a sharper result. With what I discussed earlier that sharpness can increase false positive detections.

### Otsu Thresholding

1. Pro: Otsu does not require manually setting the threshold. This technique performs best when the foreground and background have a lighting contrast. So if the object is dark and the background is light or visa versa, then Otsu typically performs well.

2. Con: Otsu does not perform well with uneven lighting or heavy amounts of shadows. Like stated earlier this typically is seen with outdoor scenes.

3. Background Noise: Since Otsu looks at the full image histogram a lot of background noise can make the binary mask noisy causing the unknown figure to merge with sections of the background.

4. Color Normalization: The increase in contrast can help Otsu make the unknown figure stand out more against the background. Since histogram equalization can enhance background texture/noise like I discussed earlier this can cause issues with the unknown figure merging with the background.

### K-Means Color Space Clustering

1. Pro: K-means does not require grayscale intensity. It uses color imformation which can be beneficial for complex outdoor scenes.

2. Cons: K-Means does not detect the cluster with the target figure. This requires the individual to inspect the clusters and choose the cluster that best captures the figure. K-Means can produce different results when using random initialization.

3. Background Noise: If the background noise is similar in color to the unknown figure than this can cause them to be grouped into the same cluster. K-Means does typically handle background noise better than Otsu since its grouping on color similarity rather than just brightness.

4. Color Normalization: Since K-Means uses color to cluster normalization can improve K-Means by making the colors more distinct.

![Segmentation Comparison](homework_two/python_script_og_image/generated_images/plots/segmentation_comparison.png)
