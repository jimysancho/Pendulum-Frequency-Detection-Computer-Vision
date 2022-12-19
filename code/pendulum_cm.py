import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from umucv.util import ROI, putText
from umucv.stream import autoStream
import pandas as pd
from scipy.ndimage import binary_fill_holes

def CM(img):
    
    n_rows, n_cols = img.shape
    rows = np.arange(0, n_rows)
    cols = np.arange(0, n_cols)
    total_mass = np.sum(img)
    # como quiero el centro en x, tengo que multiplicar por un vector fila
    x = cols[np.newaxis, :]
    # como quiero el centro en y, tengo que multiplicar por un vector columna
    y = rows[:, np.newaxis]
    
    # el centro de masas es la suma lógicamente, ya que neceistamos una coordenada
    X, Y = np.sum(img * x) / total_mass, np.sum(img * y) / total_mass

    return int(X), int(Y)
    

def apply_threshold(image, min_th=150, max_th=230, kernel=np.ones((1, 3))):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, img_th = cv.threshold(image, min_th, max_th, cv.THRESH_OTSU)
    img_th = 255 - img_th
    img_th = 255 * np.uint8(binary_fill_holes(img_th, kernel))
    return img_th


trackers = { #'BOOSTING':   cv.TrackerBoosting_create,
             'MIL':        cv.TrackerMIL_create,
             'KCF':        cv.TrackerKCF_create,
             #'TLD':        cv.TrackerTLD_create,
             #'MEDIANFLOW': cv.TrackerMedianFlow_create,
             #'GOTURN':     cv.TrackerGOTURN_create,
             #'MOSSE':      cv.TrackerMOSSE_create,
             'CSRT':       cv.TrackerCSRT_create
          }


method = 'KCF' 
videos = {
          'contraste': 'pendulo-CM.mp4', 'perturbado': 'Videos/estacionario-perturbado.mp4', 
          '1-Foucault': 'Videos/Foucault/1-intento-Foucault.mp4', '6a-Foucault': 'Videos/Foucault/6a-tanda-Foucault.mp4',
          '2a-Foucault': 'Videos/Foucault/2a-tanda-Foucault.mp4'
         }

selected_video = '1-Foucault'
cap = cv.VideoCapture(videos[selected_video])

ok = False
stop = False
NAME = 'ROI'
run = True
size = 1
points = []
pendulum = None

while run:
    
    key = cv.waitKey(1) 
    if key == 27:
        run = False
        break
        
    _, frame = cap.read()
    if frame is None:
        cap = cv.VideoCapture(videos[selected_video])
        _, frame = cap.read()
        
    big_mask = np.zeros(frame.shape[:2])
                
    #frame = cv.GaussianBlur(frame, (5, 5), 0)
    if key == ord('k'): stop = not stop
        
    if stop and not ok:
        roi = cv.selectROI(NAME, frame)
        x1, y1, x2, y2 = roi
        stop = not stop
        tracker = trackers[method]()
        tracker.init(frame,(x1,y1,x2+1,y2+1))
        ok = True
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))
           
    cv.imshow(NAME, frame)
    
    if ok:
        timer = cv.getTickCount()
        ok, (x1,y1,w,h) = tracker.update(frame)
        pendulum = frame[y1:y1 + h + 1, x1: x1 + w + 1]
        gray_pendulum = apply_threshold(pendulum, 180, 255)
        big_mask[y1: y1 + h + 1, x1: x1 + w + 1] = gray_pendulum
        mask = cv.dilate(big_mask, kernel=np.ones((3, 3)), iterations=4)
        mask = cv.erode(mask, kernel=np.ones((3, 3)), iterations=2)
        
        border = np.uint8(mask - cv.erode(mask, kernel=np.ones((3, 3))))
        border_copy = cv.dilate(border, kernel=np.ones((3, 3)))
        #frame[:, :, 1] += (border_copy // np.max(border)) * 100
        #frame[:, :, 0] += (border_copy // np.max(border)) * 100
    
        try:
            X, Y = CM(gray_pendulum)

        except:
            print('Try to select a better roi')
                    
        #cv.circle(frame, (X + x1, Y + y1), 5, (0, 255, 0), -1)
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)
        
        cv.rectangle(frame, (int(x1),int(y1)), (int(x1+w),int(y1+h)), 
                     color=(255,255,0), thickness=2)
        
        #putText(frame, f'{fps:.0f}Hz', orig=(int(x1),int(y1)-8))
        points.append([X + x1, Y + y1])
        
        #cv.circle(gray_pendulum, (X, Y), 10, (0, 255, 0), -1)
        #cv.imshow('roi', gray_pendulum * frame)
        cv.circle(frame, (X + x1, Y + y1), 8, (0, 0, 255), -1)
        cv.imshow('roi', mask)
        
        
    # para tomar capturas de pantalla
    if ok and stop:
        cv.imshow('pendulo', mask)
        cv.imshow('test', frame)
        cv.waitKey(0)
        stop = not stop
        
    putText(frame,method)
    cv.imshow(NAME,frame)