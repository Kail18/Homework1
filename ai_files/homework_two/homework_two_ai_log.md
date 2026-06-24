# All usage will be done with ChatGPT

## Prompt 1 13:34 6/24/26

1. Main project folder is Homework1 -> Venv, homework_one, homework_two, ai_files, and README.md. ai_files -> code_implementation.md, reasearch_ai.md. homework_one -> homework_one.py, myImage.png. homework_two -> homework_two.py, myImage.png

2. This was me giving information to the ai about how I had set up my project.

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
