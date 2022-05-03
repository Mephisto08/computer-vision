import sys
import cv2 as cv
import numpy as np

muenzeSize = 2.325
 
def showAndWait(img):
    cv.namedWindow("Image", cv.WINDOW_NORMAL)
    cv.resizeWindow("Image", (500, 700))
    cv.imshow("Image", img)
    cv.waitKey(0)

def main(argv):
    
    default_file = 'messerMuenze.jpg'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    showAndWait(src)
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7, 7), 0)
    showAndWait(gray)
    
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=50, maxRadius=200 )
    
    print("Circles erkannt: ")
    print(circles)

    durchmesser = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            print("sdf", i)
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            durchmesser = radius * 2
            cv.circle(src, center, radius, (255, 0, 255), 3)
    showAndWait(src)


    print("Durchmesser", durchmesser)
    pixelsPerMetric = durchmesser/muenzeSize

    print("pixelsPerMetric", pixelsPerMetric)
    
    return 0
if __name__ == "__main__":
    main(sys.argv[1:])