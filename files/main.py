import imutils
import cv2

def showAndWait(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, (500, 700))
    cv2.imshow(name, img)
    cv2.waitKey(0) 

image = cv2.imread("messerMuenze.jpg")
showAndWait("Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
showAndWait("Image", gray)

gray = cv2.GaussianBlur(gray, (5, 5), 0)
showAndWait("Image", gray)

edged = cv2.Canny(gray, 50, 100)
showAndWait("Image", edged)
edged = cv2.dilate(edged, None, iterations=1)
showAndWait("dilate 1", edged)
edged = cv2.dilate(edged, None, iterations=1)
showAndWait("dilate 2", edged)

edged = cv2.dilate(edged, None, iterations=1)
showAndWait("dilate 3", edged)
edged = cv2.erode(edged, None, iterations=1)
showAndWait("erode 2", edged)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grabcontours(cnts)
# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, ) = imutils.contours.sort_contours(cnts)
pixelsPerMetric = None