import sys
import cv2
import numpy as np

muenzeSize = 2.325
 
def showAndWait(img):
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", (500, 900))
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def calcLength(box, pixelsPerMetric):
    """
    Berechnet die Länge eines Rechtecks. Hierfür wird aus beiden Seiten der Durschschnitt berechnet.
    """
    length = (abs(box[0][1] - box[2][1])/pixelsPerMetric + abs(box[1][1] - box[3][1])/pixelsPerMetric) / 2
    return length

def calcWitdh(box, pixelsPerMetric):
    """
    Berechnet die Breite eines Rechtecks. Hierfür wird aus beiden Seiten der Durschschnitt berechnet.
    """
    witdh = (abs(box[0][0] - box[2][0])/pixelsPerMetric + abs(box[1][0] - box[3][0])/pixelsPerMetric) / 2
    return witdh

def main(argv):
    
    default_file = 'messerMuenze2.jpg'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Fehler beim lesen des Bildpfades!')
        return -1
    showAndWait(src)
    
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    showAndWait(gray)

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
            print("Kreis: ", i)
            center = (i[0], i[1])
            # circle center
            cv2.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2] 
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

    filterdBoxContours = []
    for i, c in enumerate(contours):
        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)

        witdh = calcWitdh(box, pixelsPerMetric)
        length = calcLength(box, pixelsPerMetric)
        print(witdh)
        print(length )
        #print("breite", witdh)
        #print("länge", length)

        # löscht das Rechteck aus was um das gesamte Bild gezeichnet wird
        # hierfür wird überprüft, ob eine Korrdinate 0/0 entspricht
        if (box[0][0] == 0 and box[0][1] == 0) \
                or (box[1][0] == 0 and box[1][1] == 0)\
                or (box[2][0] == 0 and box[2][1] == 0)\
                or (box[3][0] == 0 and box[3][1] == 0):
            continue
        #prüft, ob die breite eines Rechtecks mindestens x cm aufweist
        if witdh <= 1.0:
            continue
        # prüft, ob die länge des rechtecks mindestens x cm lang ist
        if length <= 10.0:
            continue

        filterdBoxContours.append(box)
    showAndWait(src)

    obj = None
    for i, box in enumerate(filterdBoxContours):
        print(box)
        # TODO: Wenn mehr als ein Element erkannt wird mit den Größen 1cm breite 10 cm länge
        obj = box
        cv2.drawContours(src, [box], 0, (0, 255, 255), 10)
 
    showAndWait(src)

    length = calcLength(box, pixelsPerMetric)
    witdh = calcWitdh(box, pixelsPerMetric)

    print("Länge des Messers: ",   length) 
    print("Breite des Messers: ",   witdh) 

    return 0
    
if __name__ == "__main__":
    main(sys.argv[1:])