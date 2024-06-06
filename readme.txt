Pattern Recognition and Computer Vision
CS5330

Final Project: Lane Detection using OpenCV: A video based approach


Created:Keval Visaria 
	Chirag Dhoka Jain
	Tejasri Kasturi
	Veditha Gudapati


------- Traditional Lane Detection Methods ------------------

laneHSL.py  ---> script processes video footage to detect and visualize lanes on the road using HSL color space. 
laneHSV.py  ---> script processes video footage to detect and visualize lanes on the road using HSV color space. 
laneRGB.py  ---> script processes video footage to detect and visualize lanes on the road using RGB color space. 

------- Advanced Lane Detection Systems -----------------------


main.py 	----> script performs lane detection on an input image. It loads an image, identifies lane markings using computer vision techniques, and 		      then applies various transformation.
edge_detection.py --> script contains functions for image processing tasks such asbinary thresholding, Gaussian blurring, Sobel edge detection, and applying 	              thresholds to image channels. These functions are designed to preprocess images before further analysis, such as lane detection.
lane_detection.py --> This script implements a class called Lane for performing lane detection on images. It utilizes several computer vision techniques 		      such as adaptive thresholding, perspective transformation, the sliding window method for lane line detection, and curvature                 		      calculation.


------- Lane Detection using CNN - TuSimple Dataset -------------

unet.py      ----> script builds U-Net Network 
load_model.py ---> script loads model from TuSimple Dataset
train_model   ---> script trains the Network
visual.py    ----> script helps us visualize the TuSimple Dataset
unet_lane_detection_final.h5   -----> model file


-----------------------------------------------------------------

Operating system: Windows 11
IDE: Visual Studio Code

-----------------------------------------------------------------

Link: https://drive.google.com/drive/folders/1Qiwvpa5qLDQ2hbbgN9DSHkag8PlwTku7