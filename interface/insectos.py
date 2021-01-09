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




""" ----------------------------------------------------------------------------------------------------------------------------------------------
    ----------------------------------- 2. Implementation of the class and its methods. ----------------------------------------------------------
    ---------------------------------------------------------------------------------------------------------------------------------------------- """
qtCreatorFile = "Interfaz.ui"                                       # Name of the GUI created using the Qt designer
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)          # The .ui file is imported to generate the graphical interface

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
        self.lb_udea.setPixmap(QtGui.QPixmap('LogoUdea.png'))       # Logo UdeA

        self.Importar.clicked.connect(self.Import_image)            # Connection of the button with the Import_image method call
        self.Start.clicked.connect(self.Start_execution)            # Connection of the button with the Start_execution method call
        self.Exit.clicked.connect(self.Exit_execution)              # Connection of the button with the Exit_execution method call
        self.setWindowIcon(QtGui.QIcon('udea.ico'))                 # Logo assignment (University's logo)

    def Start_execution(self):	
	    print('started')

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
        self.lb_Original.setPixmap(frame_original) 
        # Initialization of the variables each time a video is imported

                        
    """------------------------------------------------ g.  Stopping the execution  --------------------------------------------------------"""       
    def Stop_execution(self):
        """ This method has no input parameters and is responsiblefor partially stopping frame capture by stopping the timer that controls
            this process. Also, this method stores the results obtained so far in an Excel file."""
            
        self.Close_excel_file()     # The method for storing the statistical results in the Excel file is invoked
        self.timer_1.stop()         # The timer is stopped
        self.flag_stop = True       # The flag indicates that the execution was stopped
        

    """------------------------------------------------ h. Exiting the execution  --------------------------------------------------------"""
    def Exit_execution(self):
        """ This method has no input parameters and is responsible for definitely stopping frame capture by stopping the timer that controls 
            this process and exiting the MainWindow. Besides, this method stores the final results obtained in an Excel file and release the
            video input."""
            
        try:
            self.video.release()     # The video object is released
            self.Close_excel_file()  # The method for storing the statistical results in the Excel file is invoked
        except:
            pass
        
        self.timer_1.stop()         # The timer is stopped
        plt.close()                 # The graph of the heart expansion is closed
        window.close()              # The graphical interface is closed

# Main implementation      
if __name__ == "__main__":
    
    dirname = os.path.dirname(PyQt5.__file__)
    plugin_path = os.path.join(dirname, 'plugins', 'platforms')
    QtWidgets.QApplication.addLibraryPath(plugin_path)
    app =  QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    quit(app.exec_())
    

