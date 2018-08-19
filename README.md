# imageDetection
Image detection using opencv, Haar Cascade Classifier, Python Object oriented Programming. Every Object whether living or non-living can be identified in image detection by tweeking enough parameters in the respective algorithms and performing set of instructions as automation.
Current code is only for Face detection.

# Face Detection
Face detection/image detection has become one of the hot topics in recent decade. Thank to all open sourced algorithms, now people are able to implement these in more use cases. Facebook, Google and other companies which are able to tag Faces ais one of the best example. Remember that we use to train these algorithms ourselves by tagging our friends on respective faces.

# Face Detection in Phone
We train our face initially when we buy phone for 5-10 times. The algorithm inside the phone detects the face and then multiply these images in different ways including the facial features, then trains these images. Now it matches our face when we test again, if its able to identify with enough confidence it unlocks, else it rejects. 

Remeber there is also automation involved in all this situations, its just not the algorithm. So it is important to know programming , understand how your use case should work.

# OpenCV
OpenCV is an open source computer vision and machine learning library built using C/C++. It is a BSD-licence product thus free for both business and academic purposes. We can perform or use the openCV algorithms in applications viz; face detection, object recognition, extracting 3D models, image processing, camera calibration, anomaly detections in movements/motion etc. These openCV algorithms does have backend statistical way for image detection, dilation, image duplication/multiplication (making copies), converts images/videos to colored format and vice versa.

OpenCV suppoorts C++, C, Python and Java languages which can run on Windows, Linux, Mac OS, iOS, and Android. timized C/C++, the library can take advantage of multi-core processing.

For more about [openCV](https://docs.opencv.org/3.4.1/index.html)

## Different Classifiers available
Why are we using classifier? To understand if the object is face or not, we need to grade/classify the objects detected.. There are are two types of classifiers by openCV:
- HAAR Classifier
- LBP Classifier
- Hogcascade Classifier
These classifiers process images in gray scales, basically because we don't need color information to decide if a picture has a face or not (we'll talk more about this later on). As these are pre-trained in OpenCV, their learned knowledge files also come bundled with OpenCV [--opencv/data/--](https://github.com/opencv/opencv/tree/master/data)

OpenCV comes with a trainer as well as detector. If you want to train your own classifier for any object like car, planes etc. you can use OpenCV to create one. [Training Customization](https://docs.opencv.org/trunk/dc/d88/tutorial_traincascade.html)

## 1. HAAR Classifier
This uses **Adaboost** algorithm mainly while training and detection of the images based on several features available.

AdaBoost is a training process for face detection, which selects only those features known to improve the classification (face/non-face) accuracy of our classifier.
In the end, the algorithm considers the fact that generally: most of the region in an image is a non-face region. Considering this, it’s a better idea to have a simple method to check if a window is a non-face region, and if it's not, discard it right away and don’t process it again. So we can focus mostly on the area where a face is

## 2. LBP Classifier
Local Binary Patterns Cascade, also needs to be trained on hundreds of images. LBP is a visual/texture descriptor, and thankfully, our faces are also composed of micro visual patterns.
So, LBP features are extracted to form a feature vector that classifies a face from a non-face.
For each block, LBP looks at 9 pixels (3×3 window) at a time, and with a particular interest in the pixel located in the center of the window.

Then, it compares the central pixel value with every neighbor's pixel value under the 3×3 window. For each neighbor pixel that is greater than or equal to the center pixel, it sets its value to 1, and for the others, it sets them to 0.

After that, it reads the updated pixel values (which can be either 0 or 1) in a clockwise order and forms a binary number. Next, it converts the binary number into a decimal number, and that decimal number is the new value of the center pixel. We do this for every pixel in a block.

Then, it converts each block values into a histogram, so now we have gotten one histogram for each block in an image.

Finally, it concatenates these block histograms to form a one feature vector for one image, which contains all the features we are interested. So, this is how we extract LBP features from a picture.

Comparision:


For more about Open Cv and respective algorithms
[openCV algorithms](https://github.com/opencv/opencv.git) as xml

