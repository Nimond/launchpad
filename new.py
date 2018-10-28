import numpy as np
import cv2
import urllib.request
from playsound import playsound

hsv_min = (100, 120, 35)
hsv_max = (180, 255, 255)

f_min = (20, 50, 170)
f_max = (180, 255, 255)
cap = cv2.VideoCapture(0)

url='http://10.71.171.185:8080/shot.jpg'

while(True):
    # Capture frame-by-frame
    x = None
    y = None
    lx = None
    ly = None
    fx = None
    fy = None
    
    ret, img = cap.read()
    #imgResp=urllib.request.urlopen(url)
    #imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    #img=cv2.imdecode(imgNp,-1)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    tresh = cv2.inRange(hsv, hsv_min, hsv_max) 
    _, cnts, hierarchy = cv2.findContours(tresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0]
    for c in cnts:
        moments = cv2.moments(c)
        dM01 = moments['m01']
        dM10 = moments['m10']
        dArea = moments['m00']

        if dArea>100:
            lx = x
            ly = y
            x = int(dM10/dArea)
            y = int(dM01/dArea)
            cv2.circle(img, (x,y), 10, (0,0,255), -1)

    finger = cv2.inRange(hsv, f_min, f_max)
    _, cnts, hierarchy = cv2.findContours(finger.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        moments = cv2.moments(c)
        dM01 = moments['m01']
        dM10 = moments['m10']
        dArea = moments['m00']

        if dArea>100:
            fx = int(dM10/dArea)
            fy = int(dM01/dArea)
            cv2.circle(img, (fx,fy), 10, (0,255,0), -1)

    if x != None and fx != None and lx != None:
        if fx<abs(x-lx)/2: 
            playsound('1.mp3')
        else: 
            playsound('2.mp3')
    # Display the resulting frame
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
