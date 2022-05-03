import sys
import cv2
import numpy as np

muenzeSize = 2.325
 
def showAndWait(img):
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", (500, 700))
    cv2.imshow("Image", img)
    cv2.waitKey(0)

def main(argv):
    
    default_file = 'messerMuenze2.jpg'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    showAndWait(src)
    
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    showAndWait(gray)
    
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
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
            cv2.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            durchmesser = radius * 2
            cv2.circle(src, center, radius, (255, 0, 255), 3)
    showAndWait(src)


    print("Durchmesser", durchmesser)
    pixelsPerMetric = durchmesser/muenzeSize

    print("pixelsPerMetric", pixelsPerMetric)                     
        
    ret, threshed_img = cv2.threshold(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY),
                    127, 255, cv2.THRESH_BINARY)

    threshed_img = cv2.erode(threshed_img, None, iterations=3)
    threshed_img = cv2.dilate(threshed_img,None,iterations=3)

    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    obj = None
    for i, c in enumerate(contours):
        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        if i == 14:
            obj = box
            cv2.drawContours(src, [box], 0, (0, 255, 255), 10)
        else:
            cv2.drawContours(src, [box], 0, (0, 0, 255), 10)
 
    print("Konturen", len(contours))
    showAndWait(src)

    print("objecgt", obj) 
    length = obj[2][1] - obj[1][1] 
    length = length/pixelsPerMetric
    print(length) 

    return 0
    
if __name__ == "__main__":
    main(sys.argv[1:])