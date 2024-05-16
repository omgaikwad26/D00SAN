 
import cv2  as cv
import numpy as np 


image = cv.imread('chess.jpeg') 

# convert the input image into 
# grayscale color space 
operatedImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 

# modify the data type 
# setting to 32-bit floating point 
operatedImage = np.float32(operatedImage) 

# to detect the corners with appropriate 
#values as input parameters 
dest = cv.cornerHarris(operatedImage, 2, 5, 0.07) 

# Results are marked through the dilated corners 
dest = cv.dilate(dest, None) 

# Reverting back to the original image, 
image[dest > 0.01 * dest.max()]=[0, 0, 255] 

# the window showing output image with corners 
cv.imshow('Image with Borders', image) 

# De-allocate any associated memory usage 
if cv.waitKey(0) & 0xff == 27: 
	cv.destroyAllWindows() 

