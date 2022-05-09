import sys
import cv2
import numpy as np
import math
#from scipy.__config__ import show
from scipy.spatial import distance as dist
import pathlib
import os

muenzeSize = 2.3
 
def showAndWait(img):
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", (500, 700))
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def calcLength(box, pixelsPerMetric):
    """
    Berechnet die Länge eines Rechtecks. Hierfür wird aus beiden Seiten der Durschschnitt berechnet.
    Berechnugn wird über die Diagonaleneckpunkte berechnet.
    """
    try:
        length = (abs(box[0][1] - box[2][1])/pixelsPerMetric + abs(box[1][1] - box[3][1])/pixelsPerMetric) / 2
    except:
        return None
    return length

def calcWitdh(box, pixelsPerMetric):
    """
    Berechnet die Breite eines Rechtecks. Hierfür wird aus beiden Seiten der Durschschnitt berechnet.
    Berechnugn wird über die Diagonaleneckpunkte berechnet.
    """
    try:
        witdh = (abs(box[0][0] - box[2][0])/pixelsPerMetric + abs(box[1][0] - box[3][0])/pixelsPerMetric) / 2
    except:
        return None
    return witdh

def midpoint(ptA, ptB):
    return ((ptA[0]+ptB[0])* 0.5, (ptA[1] + ptB[1]) * 0.5)

def euclideanDist(box, pixelsPerMetric):
    (tl,tr,br,bl) = box

    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    dA = dist.euclidean((tltrX,tltrY), (blbrX,blbrY))
    dB = dist.euclidean((tlblX,tlblY), (trbrX,trbrY))

    try:
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric
    except:
        return (None, None)

    return (dimB, dimA)

def calculateLength(src, name):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    #showAndWait(gray)

    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=100, maxRadius=200 )

    durchmesser = 0
    if circles is not None:
        # hier muss durchmesser berechnet werden, um ungenauigkeit von radius zu vermeiden, da er in int gecastet wird
        durchmesser = circles[0][0][2] * 2
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2] 
            cv2.circle(src, center, radius, (255, 0, 255), 3)
    #showAndWait(src)


    #print("Durchmesser", durchmesser)
    pixelsPerMetric = durchmesser/muenzeSize

    #print("pixelsPerMetric", pixelsPerMetric)                     
        
    ret, threshed_img = cv2.threshold(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY),
                    127, 255, cv2.THRESH_BINARY)

    threshed_img = cv2.erode(threshed_img, None, iterations=3)
    threshed_img = cv2.dilate(threshed_img,None,iterations=3)

    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filterdBoxContours = []
    for i, c in enumerate(contours):
        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        (length,width) = euclideanDist(box,pixelsPerMetric)
        
        if length is None or width is None:
            continue
        # löscht das Rechteck aus was um das gesamte Bild gezeichnet wird
        # hierfür wird überprüft, ob eine Korrdinate 0/0 entspricht
        if (box[0][0] == 0 and box[0][1] == 0) \
                or (box[1][0] == 0 and box[1][1] == 0)\
                or (box[2][0] == 0 and box[2][1] == 0)\
                or (box[3][0] == 0 and box[3][1] == 0):
            continue
        #prüft, ob die breite eines Rechtecks mindestens x cm aufweist
        if width <= 1.0 and length <= 1.0:
            continue
        # prüft, ob die länge des rechtecks mindestens x cm lang ist
        if length <= 10.0 and width <= 10.0:
            continue
        #print(box)

        filterdBoxContours.append(box)
    #showAndWait(src)

    obj = None
    for i, box in enumerate(filterdBoxContours):
        # TODO: Wenn mehr als ein Element erkannt wird mit den Größen 1cm breite 10 cm länge
        obj = box
        cv2.drawContours(src, [box], 0, (0, 255, 255), 10)


    #length = calcLength(box, pixelsPerMetric)
    #witdh = calcWitdh(box, pixelsPerMetric)

    (length,width) = euclideanDist(box,pixelsPerMetric)
    
    print("")
    print("Name des Bildes: ", name)
    print("Länge des Messers: ", length)
    print("Breite des Messers: ", width)

    #howAndWait(src)

    return (length, width)


def main(argv):    
    singleData = False

    if singleData:
        default_file = 'data_test\IMG_5511.jpeg'
        filename = argv[0] if len(argv) > 0 else default_file
        src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_COLOR)
        test = calculateLength(src)
    else:
        yourpath = 'data'
        #default_file = 'computer-vision\data\IMG_5352.jpeg'

        midLW = []
        for root, dirs, files in os.walk(yourpath, topdown=False):
            for name in files:
                src = cv2.imread(cv2.samples.findFile(yourpath+'\\'+ name), cv2.IMREAD_COLOR)
                midLW.append(calculateLength(src, name))

        midL = 0.0
        midLA = []
        midW = 0.0
        midWA = []

        for e in midLW:
                # entfernt None Objekte
                if e[0] is None or e[1] is None:
                    continue
                
                # Prüft, ob Breite und Länge vertauscht sind. Fügt längeres Objekt zu länge hinzu und kleineres zur Breite
                if e[1] > e[0]:
                    midL += e[1]
                    midW += e[0]
                    midLA.append(e[1])
                    midWA.append(e[0])
                else:
                    midL += e[0]
                    midW += e[1]
                    midLA.append(e[0])
                    midWA.append(e[1])


        midL = midL / len(midLW)
        midW = midW / len(midLW)

        print(midL)
        print(midW)
        for i, e in enumerate(midLA):
            print("Länge: ", e, " , Breite: ", midWA[i])

if __name__ == "__main__":
    main(sys.argv[1:])