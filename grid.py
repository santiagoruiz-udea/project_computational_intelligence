import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from skimage.measure import label, regionprops
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split 
import cv2

nn = joblib.load('./Implementación_RNA/modelo_entrenado.pkl') # Carga del modelo.
df = pd.read_excel('./Implementación_RNA/Clasificacion.xlsx')    #leectura de datos

L, a, b, y_esperada = df["L"].values, df["A"].values, df["B"].values, df["Clase"].values
y_esperada[y_esperada == 0] = -1

X = np.transpose(np.array([L, b]))
Y = y_esperada
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

print(nn.score(X_test,Y_test))


img = cv2.imread('prueba.jpg')
Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L,a,b = cv2.split(Lab)
w = 64
h = 48
W,H,Z = img.shape

ret,imagen_binarizada = cv2.threshold(L,60,255,cv2.THRESH_BINARY_INV)
copy_imagen_binarizada = imagen_binarizada.copy()

mask_afilter = np.zeros_like(imagen_binarizada)
img_etiqueta, num_etiqueta = label(imagen_binarizada, connectivity=2,return_num=True)
region = regionprops(img_etiqueta)

for prop in region:
    if prop.area >= 1500 or prop.area <= 50:
        x_c, y_c, w_c, h_c = prop.bbox
        mask_afilter[x_c:w_c,y_c:h_c]= mask_afilter[x_c:w_c,y_c:h_c] | imagen_binarizada[x_c:w_c,y_c:h_c]

imagen_binarizada = imagen_binarizada - mask_afilter

img_etiqueta, num_etiqueta = label(imagen_binarizada, connectivity=2,return_num=True)
region = regionprops(img_etiqueta)

#limitacion del recuadro que contiene el mosquito   
for prop in region:
    x_c, y_c, w_c, h_c = prop.bbox
    try:
        mos = cv2.resize(img[x_c-10:w_c+10,y_c-10:h_c+10,:], None,fx=5 , fy=5)
    except:
        mos = cv2.resize(img[x_c:w_c,y_c:h_c,:], None,fx=5 , fy=5)
         
    cv2.imshow('mosquito', mos) 
    
    w, h = L[x_c:w_c,y_c:h_c].shape
    L_mean = np.sum(np.sum(L[x_c:w_c,y_c:h_c]))/(w*h)
    B_mean = np.sum(np.sum(b[x_c:w_c,y_c:h_c]))/(w*h)
    
    X_data = np.array([L_mean, B_mean]).reshape((1,2))
    probabilidad = nn.predict_proba(X_data)
    if probabilidad[0][1] >= 0.6:
        print('Es un mosquito con probabilidad de ' + str(probabilidad[0][1]))
    else:
        print('No es mosquito')
        
    k = cv2.waitKey(0)
    
    if k == 27: 
        cv2.destroyAllWindows()
        break


#for i in range (0, W//w):
#    for j in range (0, H//h):
#        
#        cv2.imshow('Fondo',img[w*i:w*(i+1),h*j:h*(j+1)])
#        
#        L_mean = np.sum(np.sum(L[w*i:w*(i+1),h*j:h*(j+1)]))/(w*h)
#        B_mean = np.sum(np.sum(b[w*i:w*(i+1),h*j:h*(j+1)]))/(w*h)
#        X_data = np.array([L_mean, B_mean]).reshape((1,2))
#        probabilidad = nn.predict_proba(X_data)
#        if probabilidad[0][1] >= 0.6:
#            print('Es un mosquito con probabilidad de ' + str(probabilidad[0][1]))
#        else:
#            print('No es mosquito')
#            
#        k = cv2.waitKey(0)
#        if k == 27: 
#            cv2.destroyAllWindows()
#            break