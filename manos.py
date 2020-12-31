import cv2
import numpy as np
#Redimenzionar lo fotogramas
import imutils
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
#Sustraccion del fondo y la imagen de primer plano
bg = None
# COLORES PARA VISUALIZACIÓN
color_start = (204,204,0)
color_end = (204,0,204)
color_far = (255,0,0)
color_start_far = (204,204,0)
color_far_end = (204,0,204)
color_start_end = (0,255,255)
color_contorno = (0,255,0)
color_ymin = (0,130,255) # Punto más alto del contorno
#color_angulo = (0,255,255)
#color_d = (0,255,255)
color_fingers = (0,255,255)
while True:
  ret, frame = cap.read()
  if ret == False: break  
  # Redimensionar la imagen para que tenga un ancho de 1000
  frame = imutils.resize(frame,width=900)     
  frame = cv2.flip(frame,1)
  frameAux = frame.copy()
  
  if bg is not None:

    # Determinar la región de interés
    ROI = frame[50:500,380:800]
    cv2.rectangle(frame,(380-2,50-2),(800+2,500+2),color_fingers,1)
    grayROI = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)

    # Región de interés del fondo de la imagen
    bgROI = bg[50:500,380:800]

  

    # Determinar la imagen binaria (background vs foreground)
    dif = cv2.absdiff(grayROI, bgROI)
    _, th = cv2.threshold(dif, 30, 255, cv2.THRESH_BINARY)
    th = cv2.medianBlur(th, 7)
    
    # Encontrando los contornos de la imagen binaria
    cnts, _ = cv2.findContours(th,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:1]
    for cnt in cnts:
      # Encontrar el centro del contorno
      M = cv2.moments(cnt)
      if M["m00"] == 0: M["m00"]=1
      x = int(M["m10"]/M["m00"])
      y = int(M["m01"]/M["m00"])
      cv2.circle(ROI,tuple([x,y]),5,(0,255,0),-1)
      # Punto más alto del contorno
      ymin = cnt.min(axis=1)
      cv2.circle(ROI,tuple(ymin[0]),5,color_ymin,-1)
      # Contorno encontrado a través de cv2.convexHull
      hull1 = cv2.convexHull(cnt)
      cv2.drawContours(ROI,[hull1],0,color_contorno,2)
      # Defectos convexos
      hull2 = cv2.convexHull(cnt,returnPoints=False)
      defects = cv2.convexityDefects(cnt,hull2)
      

    cv2.imshow('th',th)
  cv2.imshow('Frame',frame)
  k = cv2.waitKey(20)
  if k == ord('i'):
    bg = cv2.cvtColor(frameAux,cv2.COLOR_BGR2GRAY)
  if k == 27:
    break
cap.release()
cv2.destroyAllWindows()