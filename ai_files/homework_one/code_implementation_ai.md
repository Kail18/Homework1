i. Each entry should include the full prompt, the date and time the prompt was issued, the AI tool used to enter the prompt, a synopsis of the result, and any relevant design or code changes affected by the result.

# All usage will be done with ChatGPT

## Prompt 1 11:43am 6/12

1. Prompt: How to convert and save the image to greyscale, binary, and different color spaces (HSV, CIELAB, and HLS) using python and openCV?
2. For this one I did not want to create a seperate class since I assume the different images will be needed for later questions. I followed the AI output to convert the images since this is pretty straight forward.

## Prompt 2 12:00pm 6/12

1. how to normalize the lighting by performing histogram equalization across the V (value) channel.

2. It showed me code implementation. It also broke down the steps and why we are implementing the code. I used code similar to what the AI output.

## Prompt 3 5:46pm 6/12

1.  def apply_prewitt_detection(self):
    new_array = []
    for count, img in enumerate(self.imgArray):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    prewitt_x = cv.filter2D(img, -1, kernelx)
    prewitt_y = cv.filter2D(img, -1, kernely)
    prewitt_combined = cv.magnitude(prewitt_x, prewitt_y)
    cv.imwrite(f'prewitt_combined{count}.png', prewitt_combined)
    new_array.append(prewitt_combined)
    return new_array

    Traceback (most recent call last):
    File "/Users/kailmcguire/Desktop/WSU Computer Science/CS-898(Image Analysis and Comp Vision)/Homework1/homework_one.py", line 282, in <module>
    main()

    ```^^
    File "/Users/kailmcguire/Desktop/WSU Computer Science/CS-898(Image Analysis and Comp Vision)/Homework1/homework_one.py", line 266, in main
    first_prewitt = first_detection.apply_prewitt_detection()
    File "/Users/kailmcguire/Desktop/WSU Computer Science/CS-898(Image Analysis and Comp Vision)/Homework1/homework_one.py", line 131, in apply_prewitt_detection
    prewitt_combined = cv.magnitude(prewitt_x, prewitt_y)
    cv2.error: OpenCV(4.13.0) /Users/xperience/GHA-OpenCV-Python/\_work/opencv-python/opencv-python/opencv/modules/core/src/mathfuncs.cpp:154: error: (-215:Assertion failed) src1.size() == src2.size() && type == src2.type() && (depth == CV_32F || depth == CV_64F) in function 'magnitude'\

    give me an updated prewitt

    ```

2.  def apply_prewitt_detection(self):
    new_array = []

    for count, img in enumerate(self.imgArray):

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

        cv.imwrite(f'prewitt_combined{count}.png', prewitt_display)

        new_array.append(prewitt_display)

    return new_array

# Prompt 4 11:26 6/14

1. I have already done the image generation I just need to plotting and the README copies.

2. It gave me an updated python script using matplot lib

3. I implemented this code into the project.
