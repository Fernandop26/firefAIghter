import numpy as np
import cv2
import time
#import matplotlib.pyplot as plt
from skimage import io
#video_capture = cv2.VideoCapture(-1)
#video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 426)
#video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
#video_capture.set(3, 426)
#video_capture.set(4, 240)
def map(x, in_min, in_max, out_min, out_max):
    return int((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)

## Codigo que procresa el seguidor de linea. Se le pasa como parametro el frame
def seguirLinea(frame):
    
    kernel = np.ones((5,5),np.uint8)
    bif=0
    sec=0
    # Corta la imagen para quedarse con la parte de abajo e ignorar lo que no sea suelo
    alto=int(len(frame)/3)
    ancho=int(len(frame[0]))
    crop_img = frame[len(frame)-alto:len(frame), :]
    # Convertir a escala de grises
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    # Aplicar un desenfoque gaussiano
    blur = cv2.GaussianBlur(gray,(15,15),0)
    # Binarizar la imagen con un threshold
    ret,thresh = cv2.threshold(blur,80,255,cv2.THRESH_BINARY_INV)
    ## Aplicar modificaciones morfologicas para encontrar unicamente el contorno de la linea
    # Aplicamos open, erosion y dilation.
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    erosion = cv2.erode(opening,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    # Una vez aplicada las modificaciones encontramos los contornos
    contours,hierarchy = cv2.findContours(dilation.copy(), 1, cv2.CHAIN_APPROX_NONE)
    crop_img2=crop_img.copy()
    bif=0
    sec=0
    maped=0
    if len(contours) > 0:
        # Encontramos el contorno maximo (linea)
        c = max(contours, key=cv2.contourArea)
        cx = 0
        cy = 0
        # Aplicamos moments para encontrar el centro de masa de la linea
        try:
            M = cv2.moments(c)
            cx = int(M['m10']/(M['m00']))
            cy = int(M['m01']/(M['m00']))
        except ZeroDivisionError:
            print ('division 0')
        
        # En la imagen dibujamos donde esta el centro de masa y todos los contornos.
        cv2.line(crop_img,(cx,0),(cx,alto),(0,0,255),1)
        cv2.line(crop_img,(0,cy),(ancho,cy),(0,0,255),1)
        cv2.drawContours(crop_img, contours, -1, (0,255,0), 1)
        
        # Mapeamos la posicion horizontal del centro de masa de la linea entre -100 y 100 para saber cuanto esta separado del centro del robot y asi corregir ese fallo
        mappedCx = map(cx,0,426,-100,100)
        maped=mappedCx
        print("Cx: "+str(cx)+" mappedCx: "+str(mappedCx))
        
        # Aqui comprobamos si hay color rojo en un rango de entre -5% y 5% del tamaño de ancho de la imagen respecto al centro de esta quedandonos nada mas que con el centro de la imagen
        # Esto nos diria si hay una bifurcacion en el caso de haber rojo
        for y in range(int(len(crop_img2)/2),alto):
            for x in range(int(len(crop_img2[0])/2)-int(len(crop_img2[0])/2*0.05),int(len(crop_img2[0])/2*0.05)+int(len(crop_img2[0])/2)):
                if (crop_img2[y,x,0]<=100).any():
                    if (crop_img2[y,x,1]<=100).any():
                        if (crop_img2[y,x,2]>=125).any():
                            bif=1
        if bif==1:
            print ("bif")
            
        # Aqui comprobamos si hay color azul en un rango de entre -5% y 5% del tamaño de ancho de la imagen respecto al centro de esta quedandonos nada mas que con el centro de la imagen
        # Esto nos diria si hay un sector en el caso de haber azul
        for y in range(int(len(crop_img2)/2),alto):
            for x in range(int(len(crop_img2[0])/2)-int(len(crop_img2[0])/2*0.05),int(len(crop_img2[0])/2*0.05)+int(len(crop_img2[0])/2)):
                if (crop_img2[y,x,0]>=125).any():
                    if (crop_img2[y,x,1]<=100).any():
                        if (crop_img2[y,x,2]<=100).any():
                            sec=1
        if sec==1:
            print ("sec")
    else:
        print ("I don't see the line")
    return crop_img, bif, sec, maped