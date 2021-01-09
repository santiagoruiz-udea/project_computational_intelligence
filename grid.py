import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('C:/Users/Maycol/Documents/UdeA/Noveno semestre/Inteligencia/Proyecto/Imagenes/Trampa (1).jpg')
w = 64
h = 48
W,H,Z = img.shape

for i in range (0, W//w):
    for j in range (0, H//h):
        
        cv2.imshow('Fondo',img[w*i:w*(i+1),h*j:h*(j+1)])
        k = cv2.waitKey(0)
        if k == 27: 
            cv2.destroyAllWindows()
            break