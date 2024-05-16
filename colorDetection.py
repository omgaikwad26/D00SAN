import numpy as np
import cv2 as cv
 
filename = 'plab.jpg'
img = cv.imread(filename)

hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

boundaries = [
	#([17, 15, 100], [50, 56, 200]),
	([80,50,50], [130,255,255]),
	#([25, 146, 190], [62, 174, 250]),
	#([103, 86, 65], [145, 133, 128])
]

# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv.inRange(hsv, lower, upper)
	output = cv.bitwise_and(img, img, mask = mask)

	# show the images
	cv.imshow("images", np.hstack([img, output]))

# convert the input image into 
# grayscale color space 
operatedImage = cv.cvtColor(output, cv.COLOR_BGR2GRAY) 

# modify the data type 
# setting to 32-bit floating point 
operatedImage = np.float32(operatedImage) 

# Detector parameters
blockSize = 2
apertureSize = 9
k = 0.01

# Detecting corners
dest = cv.cornerHarris(operatedImage, blockSize, apertureSize, k)

# Results are marked through the dilated corners 
dest = cv.dilate(dest, None) 

# Reverting back to the original image, 
output[dest > 0.01 * dest.max()]=[0, 0, 255] 

# the window showing output image with corners 
cv.imshow('Image with Borders', output) 

cv.waitKey(0)

"""
edges = cv.Canny(output,100,200)

 
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
 
plt.show()

ret, labels, stats, centroids = cv.connectedComponentsWithStats(dest)

#define the criteria to stop and refine the corners
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
#here u can get corners
print (corners)

#Now draw them
res = np.hstack((centroids,corners)) 
res = np.int0(res) 
img[res[:,1],res[:,0]]=[0,0,255] 
img[res[:,3],res[:,2]] = [0,255,0]
cv.imwrite('1.png',img)

"""
