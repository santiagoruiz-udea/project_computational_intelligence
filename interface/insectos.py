# -*- coding: utf-8 -*-
""" ----------------------------------------------------------------------------------------------------------------------------------------------
    ------------------------------------------------ GEPAR and GeoLimna research groups ----------------------------------------------------------
    ----------------------------------------------------- University of Antioquia ----------------------------------------------------------------
    ------------------------------------------------------- Medellín, Colombia -------------------------------------------------------------------
    -------------------------------------------------------- September, 2019 ---------------------------------------------------------------------
    ----------------------------------------------------------------------------------------------------------------------------------------------
    --------------------------------------------- Authors: * David Stephen Fernández Mc Cann -----------------------------------------------------
    ------------------------------------------------------ * Fabio de Jesús Vélez Macias ---------------------------------------------------------
    ------------------------------------------------------ * Nestor Jaime Aguirre Ramírez --------------------------------------------------------
    ------------------------------------------------------ * Santiago Ruiz González --------------------------------------------------------------
    ------------------------------------------------------ * Maycol Zuluaga Montoya --------------------------------------------------------------
    ----------------------------------------------------------------------------------------------------------------------------------------------
    ------------ Project Name: Estimation of the heart rate of the Daphnia pulex by applying artificial vision techniques ------------------------
    ----------------------------------------------------------------------------------------------------------------------------------------------
    ----------Description: This project aims to implement an algorithm to estimate the heart rate of the Daphnia pulex by applying ---------------
    ---------------------- artificial vision techniques and using digital images extracted from videos captured through a microscope. ------------
    ---------------------- Is asked to the user to import a video of the Daphnia pulex and the heart rate is calculated. -------------------------
    ---------------------------------------------------------------------------------------------------------------------------------------------- """
     

""" ----------------------------------------------------------------------------------------------------------------------------------------------
    --------------------------------------------- 1. Import of the necessary libraries -----------------------------------------------------------
    ---------------------------------------------------------------------------------------------------------------------------------------------- """
import os                                                           # Operating system dependent functionalities module
import cv2                                                          # Images processing module 
import sys                                                          # Module of variables and functions used by the interpreter
import math                                                         # C standard mathematical functions module
import PyQt5                                                        # Module of the set of Python bindings for Qt v5 
import matplotlib                                                   # Plotting module 
import numpy as np                                                  # Mathematical functions module
import matplotlib.pyplot as plt                                     # Module that provides a MATLAB-like plotting framework 
from xlsxwriter import Workbook                                     # Module for creating Excel XLSX files
from skimage.measure import label, regionprops                      # Additional images processing module
from PyQt5 import QtCore, QtGui, uic, QtWidgets                     # Additional elements of PyQt5 module 
from PyQt5.QtWidgets import QMessageBox                             # Module to asking the user a question and receiving an answer
from PyQt5.QtGui import QCursor
import pandas as pd
import joblib
from skimage.measure import label, regionprops
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split 
from PyQt5.QtWidgets import QPushButton,QVBoxLayout


import sys
try:
    from PyQt5.QtCore import Qt, QT_VERSION_STR
    from PyQt5.QtGui import QImage
    from PyQt5.QtWidgets import QApplication, QFileDialog
except ImportError:
    try:
        from PyQt4.QtCore import Qt, QT_VERSION_STR
        from PyQt4.QtGui import QImage, QApplication, QFileDialog
    except ImportError:
        raise ImportError("Requires PyQt5 or PyQt4.")
from QtImageViewer import QtImageViewer


""" ----------------------------------------------------------------------------------------------------------------------------------------------
    ----------------------------------- 2. Implementation of the class and its methods. ----------------------------------------------------------
    ---------------------------------------------------------------------------------------------------------------------------------------------- """
qtCreatorFile = "Interfaz.ui"                                       # Name of the GUI created using the Qt designer
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)          # The .ui file is imported to generate the graphical interface

qtCreatorFile = "results.ui"                                       # Name of the GUI created using the Qt designer
Ui_results, QtBaseClass = uic.loadUiType(qtCreatorFile)          # The .ui file is imported to generate the graphical interface

class Second(QtWidgets.QMainWindow, Ui_results):
    def __init__(self,  *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self,*args,**kwargs)
        self.setupUi(self)
        
        self.lb_udea.setMargin(3)                                   # Logo UdeA
        self.lb_gepar.setMargin(3)                                  # Logo GEPAR
        self.lb_capiro.setMargin(3)                                 # Logo Capiro
        
# Implementation of the class MainWindow
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    
    """---------------------------------------------- a. Constructor ---------------------------------------------------"""
    def __init__(self, *args, **kwargs):
        """ This method is the class constructor, where the necessary variables for the correct functioning of the software
    	    are defined as attributes. Also here, the icon is configured and the images of the logos are assigned to the
	        Labels established for it. Finally, the interface buttons are connected with their associated functions. """

        QtWidgets.QMainWindow.__init__(self,*args,**kwargs)
        self.setupUi(self)

        # Atributes
        self.image_path = 0
        self.image = 0
        self.lb_udea.setPixmap(QtGui.QPixmap('LogoUdea.png'))       # Logo UdeA
        self.lb_gepar.setPixmap(QtGui.QPixmap('LogoGepar.png'))     # Logo GEPAR
        self.lb_capiro.setPixmap(QtGui.QPixmap('LogoCapiro.png'))   # Logo Capiro
        
        self.lb_udea.setMargin(3)                                   # Logo UdeA
        self.lb_gepar.setMargin(3)                                  # Logo GEPAR
        self.lb_capiro.setMargin(3)                                 # Logo Capiro
                
        self.Importar.clicked.connect(self.Import_image)            # Connection of the button with the Import_image method call
        self.Start.clicked.connect(self.Start_execution)            # Connection of the button with the Start_execution method call
        self.Results.clicked.connect(self.Show_results)          # Connection of the button with the Show_results method call
        self.Exit.clicked.connect(self.Exit_execution)              # Connection of the button with the Exit_execution method call
        self.setWindowIcon(QtGui.QIcon('udea.ico'))                 # Logo assignment (University's logo)
        self.frame_original = QtImageViewer(self)
        self.frame_processed = QtImageViewer(self)
        
        self.frame_original.hide()
        self.frame_processed.hide()
        #Variables para calcular resultados
        self.cont_incestos = 0
        self.average_lenght = []
        self.average_width = []
        self.area = 0
        
        
    """-------------------------------------- b. Choice of directory for reading the video --------------------------------------------------- """
    def Import_image(self):
        """  In this method, the file browser is enabled so that the user selects the video that will be analyzed. Some variables are 
    	     initialized and it is started the timer that will control the capture of frames in the video to draw the line (In the preview
             of the video selected). This function doesn't return any variables."""
            
        self.image_path, _ = QtWidgets.QFileDialog.getOpenFileName(self)   # File explorer is opened to select the video that will be used                     
        self.image = cv2.imread(self.image_path)                     # A video object is created according to the path selected

        imagen_original=QtGui.QImage(self.image,self.image.shape[1],self.image.shape[0],self.image.shape[1]*self.image.shape[2],QtGui.QImage.Format_RGB888)
        frame_original = QtGui.QPixmap()
        frame_original.convertFromImage(imagen_original.rgbSwapped())

        self.frame_original.setImage(frame_original)
        self.frame_original.setGeometry(20, 50, 642, 483)
        self.frame_original.setStyleSheet("background:  rgb(39, 108, 222); border: 1px solid rgb(39, 108, 222)")
        self.frame_original.show()

    """-------------------------------------- c. Predictions --------------------------------------------------- """
    def Start_execution(self):
        nn = joblib.load('../Implementación_RNA/modelo_entrenado.pkl') # Carga del modelo.
        df = pd.read_excel('../Implementación_RNA/Clasificacion.xlsx')    #leectura de datos
        
        L, a, b, y_esperada = df["L"].values, df["A"].values, df["B"].values, df["Clase"].values
        y_esperada[y_esperada == 0] = -1
        
        X = np.transpose(np.array([L, b]))
        Y = y_esperada
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
        
        #print(nn.score(X_test,Y_test))
        
        
        img = self.image
        Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L,a,b = cv2.split(Lab)
        w = 64
        h = 48
        W,H,Z = img.shape
        
        ret,imagen_binarizada = cv2.threshold(L,50,255,cv2.THRESH_BINARY_INV)
        copy_imagen_binarizada = imagen_binarizada.copy()
        
        mask_afilter = np.zeros_like(imagen_binarizada)
        img_etiqueta, num_etiqueta = label(imagen_binarizada, connectivity=2,return_num=True)
        region = regionprops(img_etiqueta)
        font = cv2.FONT_HERSHEY_DUPLEX
        
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
                 
            #cv2.imshow('mosquito', mos) 
            
            w, h = L[x_c:w_c,y_c:h_c].shape
            L_mean = np.sum(np.sum(L[x_c:w_c,y_c:h_c]))/(w*h)
            B_mean = np.sum(np.sum(b[x_c:w_c,y_c:h_c]))/(w*h)
            
            X_data = np.array([L_mean, B_mean]).reshape((1,2))
            probabilidad = nn.predict_proba(X_data)
            if probabilidad[0][1] >= 0.75:
                self.area += np.sum(copy_imagen_binarizada[x_c:w_c,y_c:h_c])

                cv2.rectangle(img, (y_c, x_c), (h_c, w_c), (0,255,0),2)
                cv2.putText(img, 'Mosquito', (y_c, x_c - 8), font, 0.4, (0,255,0), 1,cv2.LINE_AA)
                cv2.putText(img, str(round(probabilidad[0][1],2)), (y_c + 60, x_c - 8), font, 0.4, (0,255,0), 1,cv2.LINE_AA)
                self.cont_incestos += 1
                if h_c > w_c :
                    self.average_lenght.append(h_c)
                    self.average_width.append(w_c)
                else:
                    self.average_lenght.append(w_c)
                    self.average_width.append(h_c)
                    

        cv2.imwrite(self.image_path[:-4]+'_labeled.JPG', img)
        processed = img
        imagen_processed=QtGui.QImage(processed,processed.shape[1],processed.shape[0],processed.shape[1]*processed.shape[2],QtGui.QImage.Format_RGB888)
        frame_processed = QtGui.QPixmap()
        frame_processed.convertFromImage(imagen_processed.rgbSwapped())
                
        self.frame_processed.setImage(frame_processed)
        self.frame_processed.setGeometry(700, 50, 642, 483)
        self.frame_processed.setStyleSheet("background:  rgb(39, 108, 222); border: 1px solid rgb(39, 108, 222)")
        self.frame_processed.show()
        
        self.Results.setStyleSheet("color: white; background:  rgb(39, 108, 222); border: 1px solid white; border-radius: 10px; font: 75 14pt 'Reboto Medium';")
        self.Results.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        
    """------------------------------------------------ g.  Stopping the execution  --------------------------------------------------------"""       
    def Show_results(self):
        self.dialog = Second(self)
        self.dialog.label_result.setText(str(self.cont_incestos))
        self.dialog.label_length.setText(str(round(sum(self.average_lenght)/len(self.average_lenght),2)) +' px')
        self.dialog.label_width.setText(str(round(sum(self.average_width)/len(self.average_width),2)) +' px')
        self.dialog.label_area.setText(str(self.area) +' px')
        self.dialog.show()
        self.dialog.Exit.clicked.connect(self.Exit_dialog)
        
    
         

    """------------------------------------------------ h. Exiting the execution  --------------------------------------------------------"""
    def Exit_execution(self):
        """ This method has no input parameters and is responsible for definitely stopping frame capture by stopping the timer that controls 
            this process and exiting the MainWindow. Besides, this method stores the final results obtained in an Excel file and release the
            video input."""

        window.close()              # The graphical interface is closed

    """------------------------------------------------ i. Exiting the Dialog    --------------------------------------------------------"""
    def Exit_dialog(self):
        """ This method has no input parameters and is responsible for definitely stopping frame capture by stopping the timer that controls 
            this process and exiting the Dialog. Besides, this method stores the final results obtained in an Excel file and release the
            video input."""
        self.dialog.close()

# Main implementation      
if __name__ == "__main__":
    
    dirname = os.path.dirname(PyQt5.__file__)
    plugin_path = os.path.join(dirname, 'plugins', 'platforms')
    QtWidgets.QApplication.addLibraryPath(plugin_path)
    app =  QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    

