import cv2 as cv
img = cv.imread('7229454ac33a3e86c5be5dd8d94881b8.jpg')

cv.imshow("Display window", img)
k = cv.waitKey(0) # Wait for a keystroke in the window