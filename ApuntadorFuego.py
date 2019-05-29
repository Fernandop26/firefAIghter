import cv2
def map(x, in_min, in_max, out_min, out_max):
    return int((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)

def apuntadorFuego(frame,dist):

    contours,hierarchy = cv2.findContours(frame.copy(), 1, cv2.CHAIN_APPROX_NONE)
    fire=frame.copy()
    #bif=0
    #sec=0
    # Find the biggest contour (if detected)
    #(640,480)
    mappedCx=0
    mappedCy=0
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        cx = 0
        cy = 0
        try:
            M = cv2.moments(c)
            cx = int(M['m10']/(M['m00']))
            cy = int(M['m01']/(M['m00']))
        except ZeroDivisionError:
            print ('me cago en to')
        cv2.line(fire,(cx,0),(cx,480),(0,0,255),1)
        cv2.line(fire,(0,cy),(640,cy),(0,0,255),1)
        cv2.drawContours(fire, contours, -1, (0,255,0), 1)
        mappedCx = map(cx,0,640,-85,85)
        mappedCy = map(dist,0,300,3.5,8.5)
        
        print("Cx: "+str(cx)+" mappedCx: "+str(mappedCx))
        print("Cy: "+str(cy)+" mappedCy: "+str(mappedCy))
    else:
        print ("I don't see fire")
    return fire, mappedCx, mappedCy