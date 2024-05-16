import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Read the input image
filename = 'blackPage.jpeg'
img = cv.imread(filename)
if img is None:
    print("Error: Image not found.")
    exit()
"""
# Convert the image to HSV color space
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Define the color boundaries for blue in HSV space
boundaries = [
    ([80, 50, 50], [130, 255, 255])  # Blue color range
]

# Initialize output
output = None

# Loop over the boundaries to detect specified colors
for (lower, upper) in boundaries:
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    
    # Create a mask for the color range
    mask = cv.inRange(hsv, lower, upper)
    
    # Apply the mask to the original image
    output = cv.bitwise_and(img, img, mask=mask)
"""
# Convert the masked output image to grayscale
operatedImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
operatedImage = cv.GaussianBlur(operatedImage, (5, 5), 0)

# Convert to 32-bit floating point for Harris corner detection
operatedImage = np.float32(operatedImage)

# Detector parameters
#blockSize = 5 
#apertureSize = 5
#k = 0.05

# Detect corners
#dest = cv.cornerHarris(operatedImage, blockSize, apertureSize, k)

# Dilate the corners for better visibility
#dest = cv.dilate(dest, None)

# Reverting back to the original image, 
#img[dest > 0.01 * dest.max()]=[0, 0, 255] 

#cv.imshow('Corners', img)

# Threshold to get the coordinates of the corners
#threshold = 0.01 * dest.max()
#corners = np.argwhere(dest > threshold)
corners = cv.goodFeaturesToTrack(operatedImage, maxCorners=4, qualityLevel=0.01, minDistance=10, blockSize=7)

# Ensure corners are detected
if corners is not None:
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv.circle(img, (x, y), 5, (0, 255, 0), -1)

# Plotting the image and corners using Matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display in Matplotlib

# Plot the corners
for i in corners:
    x, y = i.ravel()
    plt.plot(x, y, 'g', markersize=2)  # 'ro' means red color, 'o' shape marker

plt.title('Harris Corners')
plt.show()
