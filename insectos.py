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
        self.x_o = -1                                               # X coordinate for first point  
        self.y_o = -1                                               # Y coordinate for first point
        self.x = -1                                                 # X coordinate for second point 
        self.y = -1                                                 # Y coordinate for second point
        self.video_path=0                                           # variable that contains the path the file (video)
        self.video = 0                                              # variable that contains the video (each frame)                                               
        self.count = 0                                              # Number of the current frame
        self.flag_finish = False                                    # This flag indicates the finish of the process
        self.flag_stop = False                                      # This flag indicates that the execution was stopped
        self.flag_draw_line = False                                 # This flag indicates that there is a line  
        self.point_x = [0,0]                                        # X coordinate the current point and after point
        self.point_y = [0,0]                                        # Y coordinate the current point and after point
        self.temp = 0                                               # variable that contains the slope of the line 
        self.opening = list()                                       # list that contains the Daphnia’s heart expansion
        self.frames_list = list ()                                  # list that contains the number frame
        self.max_opening = list()                                   # list that contains the maximum Daphnia’s heart expansion
        self.timer_1 = QtCore.QTimer(self)                          # A timer is created so that when the time limit is met, take a capture from the video
        self.timer_2 = QtCore.QTimer(self)                          # A timer is created so that when the time limit is reached take a capture from the video.
        self.lb_gepar.setPixmap(QtGui.QPixmap('LogoGepar.png'))     # Logo GEPAR
        self.label.setPixmap(QtGui.QPixmap('LogoGeoLimna.png'))     # Logo GeoLimna
        self.lb_udea.setPixmap(QtGui.QPixmap('LogoUdea.png'))       # Logo UdeA

        self.Importar.clicked.connect(self.Import_video)            # Connection of the button with the Import_video method call
        self.Start.clicked.connect(self.Start_execution)            # Connection of the button with the Start_execution method call
        self.Stop.clicked.connect(self.Stop_execution)              # Connection of the button with the Stop_execution method call
        self.Exit.clicked.connect(self.Exit_execution)              # Connection of the button with the Exit_execution method call
        self.setWindowIcon(QtGui.QIcon('udea.ico'))                 # Logo assignment (University's logo)


    """-------------------------------------- b. Choice of directory for reading the video --------------------------------------------------- """
    def Import_video(self):
        """  In this method, the file browser is enabled so that the user selects the video that will be analyzed. Some variables are 
    	     initialized and it is started the timer that will control the capture of frames in the video to draw the line (In the preview
             of the video selected). This function doesn't return any variables."""
            
        self.video_path, _ = QtWidgets.QFileDialog.getOpenFileName(self)   # File explorer is opened to select the video that will be used                     
        self.video = cv2.VideoCapture(self.video_path)                     # A video object is created according to the path selected
        self.timer_2.timeout.connect(self.Draw_line)                       # timeout() signal is connected with the function Draw_line().
        self.timer_2.start()                                               # The timer that controls the preview and the draw of the line is started
        
        # Initialization of the variables each time a video is imported
        self.x_o = -1
        self.y_o = -1
        self.x = -1
        self.y = -1
        self.count = 0
        self.flag_draw_line = False
        self.max_opening = list()
        plt.ioff()
        plt.close()
        
        
    """------------------------------------------------ c. Getting x and y position  -------------------------------------------------------- """
    def mousePressEvent(self, event):
        """ This method has no input parameters and is responsible for displaying a preview of the selected video, and draws a line between the
    	    two points that the user selected (In case the user has already done so). When the line is drawn, the clock that controls the frames
            the preview view stops, so the preview also stops. This function doesn't return any variables."""
       
        if self.flag_draw_line != True:
            if self.x_o == -1:                  # The first point's coordinates are saved
                self.x_o = event.x()
                self.y_o = event.y()
            else:
                self.x = event.x()              # The second point's coordinates are saved
                self.y = event.y()
                self.flag_draw_line = True
        
        
    """--------------------------------------------------- d. Drawing of the line  --------------------------------------------------------- """
    def Draw_line(self):
        """ This method has no input parameters and is responsible for displaying a preview of the selected video, and draws a line between
            the two points that the user selected (In case the user has already done so). When the line is drawn, the clock that controls
            the frames of the preview view stops, so the preview also stops. """     
            
        try:    
            ret,frame = self.video.read()		                                                  # A new frame is captured
            frame = cv2.resize(frame, (561, 301))	                                              # The size of the frame is changed  
            
            if self.flag_draw_line == True: 
                cv2.line(frame, (self.x_o-80,self.y_o-30),(self.x-80,self.y-30), [0,255,255], 2)  # The line is drawn over the frame
                self.timer_2.stop()                                                               # The timer is stopped

            # Setting of the image of the Daphnia with the line traced
            imagen_original=QtGui.QImage(frame,frame.shape[1],frame.shape[0],frame.shape[1]*frame.shape[2],QtGui.QImage.Format_RGB888)
            frame_original = QtGui.QPixmap()
            frame_original.convertFromImage(imagen_original.rgbSwapped())
            self.lb_Original.setPixmap(frame_original) 
        except:
            pass
          
            
    """---------------------------------------------- e.  Starting the execution  ---------------------------------------------------------- """
    def Start_execution(self):
        """ This method has no input parameters and is responsible for starting the frame capture again by initializing the timer that is 
            connected to the method that performs the image processing to determine the frequency. """
            
        self.timer_1.timeout.connect(self.heart_rate_calculation)    # Timer's timeout() signal is connected with the function heart_rate_calculation().
        self.timer_1.start()                                         # The timer that controls the heart rate calculation is started
        
        if self.flag_stop == False:
            self.video = cv2.VideoCapture(self.video_path)           # If it's a new execution, the video object is created again
        else:
            self.flag_stop = False                                   # If it was stopped it will resume in the last frame 
            

    """------------------------------------------ f.  Daphnia's heart rate calculation  ------------------------------------------------------"""
    def heart_rate_calculation(self):
        """ This method which has no input parameters is the core of FECAD because this is where the processing of the captured 
            images is performed and where the heart rate estimate is made. Initially, a frame is captured, then, using the V layer
            of the image in the HSV space, a preprocessing of said frame is performed, including histogram equalization, filtering,
            morphology among others. Later an image is created that only has the line drawn and an "AND" operation is performed
            between the said image with the line and the preprocessed frame so that only the intersection of both images prevails.
            This intersection line is taken and the circle with the smallest radius surrounding 10 that line is obtained, to
            determine the length of the line (Expansion of the heart) based on the radius of this circle. The graphing of each point
            obtained is carried out taking into account that the maximum points are stored in an array because based on them and with
            the FPS of the video it is possible to obtain the estimated heart rate. """
        
        try:
            ret,frame = self.video.read()		                                 # A new frame is captured
            
            if ret and self.count == 0: 
                self.flag_finish = True                                         
            
            fps = round(self.video.get(cv2.CAP_PROP_FPS),0)                      # The number of frames per second of the video is obtained
            frame = cv2.resize(frame, (561, 301))	                             # The original frame is resized
            frame_line = frame.copy()                                            # A copy of the original frame is made
            h = abs(self.y - self.y_o)                                           # Height of the box of interest according to the line drawn
            w = abs(self.x - self.x_o)                                           # Width of the box of interest according to the line drawn
            
            if self.x < self.x_o:   pos_x = self.x - 80                          # The width of the interest box is increased 
            else:   pos_x = self.x_o - 80
 
            if self.y < self.y_o:   pos_y = self.y - 30                          # The height of the interest box is increased 
            else:   pos_y = self.y_o - 30
                
            frame_processed = frame.copy()                                       # Another copy of the original frame
            frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2HSV)   # The frame is converted to the color space HSV
            H,S,V = cv2.split(frame_processed)                                   # It is separated into the 3 layers (H,S,V)
           
            # Image preprocessing
            frame_processed = cv2.medianBlur(V, 3)                               # Smoothing of the image using the median filter
            frame_processed = cv2.equalizeHist(frame_processed)                  # An equalization of the histogram is performed.
            kernel = np.ones((3,3),np.uint8)                                     # Definition of the structural element for applying morphology
            frame_processed = cv2.erode(frame_processed,kernel,iterations = 1)   # An erosion is performed
            frame_process_BW = cv2.adaptiveThreshold(frame_processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5) # Adaptative tresholding
            frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_GRAY2BGR)                           # The frame is returned to BGR
            cv2.line(frame_processed, (self.x_o-80,self.y_o-30),(self.x-80,self.y-30), [0,255,255], 2)    # The line is drawn over the processed frame


            # Area filter discarding objects with small areas
            img_etiqueta, num_etiqueta = label(frame_process_BW, connectivity=2,return_num=True)
            region = regionprops(img_etiqueta)
            mask_afilter = np.zeros_like(frame_process_BW)
            for prop in region:
                if prop.area <= 200:
                    x_c, y_c, w_c, h_c = prop.bbox
                    mask_afilter[x_c:w_c,y_c:h_c]= mask_afilter[x_c:w_c,y_c:h_c] | frame_process_BW[x_c:w_c,y_c:h_c]
            
            # Application of the mask gotten in the area filter
            frame_process_BW = frame_process_BW - mask_afilter                                      # Small objects are removed from the image
            frame_process_BW_3layers = cv2.cvtColor(frame_process_BW,cv2.COLOR_GRAY2RGB)            # The frame is returned to BGR
            mask_line = np.zeros_like(frame)                                                        # Empty frame

            cv2.line(frame_line, (self.x_o-80,self.y_o-30),(self.x-80,self.y-30), [0,255,255], 3)   # The line is traced over the original frame
            cv2.line(mask_line, (self.x_o-80,self.y_o-30),(self.x-80,self.y-30), [255,255,255], 1)  # The line is traced over the empty frame
            
            frame_process_BW = frame_process_BW_3layers | mask_line                                 # The processed frame and the frame with the line are combined
            
            # Adecuation of frames
            frame_process_BW = frame_process_BW[pos_y-10:pos_y+h+10, pos_x-10:pos_x+w+10]                 # It's taken a mask of the frame (The line box)
            frame_processed = frame_processed[pos_y-50:pos_y+h+50, pos_x-50:pos_x+w+50]                   # It's taken a mask of the frame (The line box)
            mask_line = mask_line[pos_y-10:pos_y+h+10, pos_x-10:pos_x+w+10]                               # It's taken a mask of the frame (The line box)
            frame_process_BW_3layers = frame_process_BW_3layers[pos_y-10:pos_y+h+10, pos_x-10:pos_x+w+10] # It's taken a mask of the frame (The line box)
            frame_process_BW = cv2.resize(frame_process_BW, (561, 301))	                                  # The frame is resized
            frame_processed = cv2.resize(frame_processed, (561, 301))	                                  # The frame is resized
            mask_line = cv2.resize(mask_line, (561, 301))                                                 # The frame is resized
            frame_process_BW_3layers = cv2.resize(frame_process_BW_3layers, (561, 301))                   # The frame is resized
            
            linea_no_int = (frame_process_BW_3layers & mask_line)                                               # It's taken the line intersection in the processed frame
            linea_no_int = mask_line - linea_no_int                                                             # Original frame objects found in the line region are subtracted from the complete line
            linea_no_int=cv2.cvtColor(linea_no_int, cv2.COLOR_RGB2GRAY)                                         # The frame is converted to gray

            (a,linea_no_int_bin) = cv2.threshold(linea_no_int, 50, 255, cv2.THRESH_BINARY)                      # Thresh holding binarization
            img, contours, hierarchy = cv2.findContours(linea_no_int_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)   # Finding contours
            contours = sorted(contours, key=cv2.contourArea,reverse=True)                                       # Contours are sorted by size
            linea_no_int = contours[0]                                                                          # It's selected the biggest contour
            
            (x,y),radius = cv2.minEnclosingCircle(linea_no_int)                                                 # It's found the minimum enclosing circle (Center and radius)
            linea_no_int_bin[...] = 0                                                                           # The frame is emptied  
            cv2.drawContours(linea_no_int_bin, [linea_no_int], 0, 255, cv2.FILLED)                              # It's drawn the biggest contour     
            
            matplotlib.rcParams.update({'font.size': 12})
            ax = plt.gca()
            self.count += 1                                                                                     
            
            
            if self.count == 1:
                self.point_x[0]=self.count                                                             # It's assigned the frame number
                self.point_y[0]=(2*radius/100)                                                         # The diameter of the circle represents the heart expansion  
            else:
                self.point_x[1]=self.count                                                             # It's assigned the frame number                                                             
                self.point_y[1]=(2*radius/100)                                                         # The diameter of the circle represents the heart expansion
                ax.plot(self.point_x,self.point_y,'#2DFA18', linewidth=3)                              # Plotting of the point
                slope = ((self.point_y[1]-self.point_y[0])/(self.point_x[1]-self.point_x[0]))          # Calculation of the line's slope 
                
                if slope < 0 :
                    if  self.temp > 0:                                                      
                        max_y = self.point_y[0]                                                        # The maximum "y" value
                        self.max_opening.append(max_y)                                                 # It's assigned the maximum "y" (heart expansion) value
                        indice = self.point_y.index(max_y)                                             # The index of the maximum "y" coordinate is gotten
                        ax.plot(self.point_x[indice],max_y,'ro')                                       # It's plotted the point 

                self.opening.append(round(2*radius/100,2))                                             # It's added the new opening element
                self.frames_list.append(self.count)                                                    # It's added the new number frame
                self.temp = slope                                                                      # The slope is stored temporarily
                self.point_x[0]=self.count                                                             # It's assigned the frame number
                self.point_y[0]=(2*radius/100)                                                         # The diameter of the circle represents the heart expansion

            # Graph configurationand plotting
            ax.patch.set_facecolor('black')
            plt.grid(which='major', color='#4ECE41', linestyle='-', linewidth=1)
            plt.ylabel('Heart expansion')
            plt.xlabel('Frame number')
            plt.savefig("graph.tif",bbox_inches='tight')
            grafica_image = cv2.imread("graph.tif")
            grafica_image = cv2.resize(grafica_image, (1001,281))
                      
            font = cv2.FONT_HERSHEY_SIMPLEX                                                         # Font configuration
            seconds = round((self.count/fps),2)                                                     # Seconds calculation according to the frame rate
            latidos = 'Beats = ' +  str(len(self.max_opening)) +' in ' + str(seconds) + ' seconds'  # String to put in the interface
            cv2.rectangle(frame_line,(1,274),(345,340),(32,255,238),cv2.FILLED)                     # Rectangle to put the previous string    
            cv2.putText (frame_line, latidos , (10,291), font, 0.7, (0,0,0), 1, cv2.LINE_AA)        # The string is set over the previous rectangle    
            
            frequency = round(len(self.max_opening)*60/seconds,0)                                   # Frequency calculation according to the frame rate
            frequency = 'Beats per minute = ' +  str(frequency) +' Beats/minute'                    # String to put in the interface
            cv2.rectangle(frame_line,(1,7),(590,27),(32,255,238),cv2.FILLED)                        # Rectangle to put the previous string
            cv2.putText (frame_line, frequency , (10,24), font, 0.7, (0,0,0), 1, cv2.LINE_AA)       # The string is set over the previous rectangle
            
            # Setting of the original frame and the processed one in the interface      
            imagen_original=QtGui.QImage(frame_line,frame_line.shape[1],frame_line.shape[0],frame_line.shape[1]*frame_line.shape[2],QtGui.QImage.Format_RGB888)
            frame_original = QtGui.QPixmap()
            frame_original.convertFromImage(imagen_original.rgbSwapped())
            
            imagen_procesada=QtGui.QImage(frame_processed,frame_processed.shape[1],frame_processed.shape[0],frame_processed.shape[1]*frame_processed.shape[2],QtGui.QImage.Format_RGB888)
            frame_processed = QtGui.QPixmap()
            frame_processed.convertFromImage(imagen_procesada.rgbSwapped())
            
            # Configuration to put the graph in the graphical interface   
            grafica=QtGui.QImage(grafica_image,grafica_image.shape[1],grafica_image.shape[0],grafica_image.shape[1]*grafica_image.shape[2],QtGui.QImage.Format_RGB888)
            frame_grafica = QtGui.QPixmap()
            frame_grafica.convertFromImage(grafica.rgbSwapped())
            
            self.lb_Original.setPixmap(frame_original)
            self.lb_Procesado.setPixmap(frame_processed)
            self.lb_Original_2.setPixmap(frame_grafica)

        except:    
            self.timer_1.stop()
            
            if self.flag_finish == True:
                try:
                    # It is indicated that it's done and the results path is shown
                    message = 'The video was processed succesfully, you can see the statistical result at the\nfollowing path:\n' + self.video_path[:-4] + '.'
                    QMessageBox.setStyleSheet(self,"QMessageBox\n{\n	background-color: rgb(255, 255, 255);\n}\n")
                    QMessageBox.information(self, 'Execution finalized', message)
                except:
                    # It is indicated that it hasn't been imported a video yet
                    message = 'The video has not been imported, please do so to be able to extract information.'
                    QMessageBox.setStyleSheet(self,"QMessageBox\n{\n	background-color: rgb(255, 255, 255);\n}\n")
                    QMessageBox.critical(self, 'Start error', message)
                    self.flag_finish = False

                self.video.release()            # The video object is released
                self.Close_excel_file()         # The method for storing the statistical results in the Excel file is invoked
                
            else:
                # It is indicated that the video imported is not valid
                message = 'The file selected is invalid, please import a valid file to extract information'
                QMessageBox.setStyleSheet(self,"QMessageBox\n{\n	background-color: rgb(255, 255, 255);\n}\n")
                QMessageBox.critical(self, 'Import error', message)
               
    
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
                  
        
    """------------------------------------------- i.  Closing of the excel file  --------------------------------------------------------"""
    def Close_excel_file(self):
        """ This method has no input parameters and is responsible for store the results obtained in an Excel file. These data are statistical
            analyzes that are obtained based on the results of the execution, such as the maximum opening of the heart, the estimated maximum
            frequency, the standard deviation of the opening, among others. A folder with the same name of the video is generated and within it,
            an excel file is generated where all the information is entered. """
            
        try:
            # Definition of the variables that will be saved
            maximum_opening = max(self.opening)                                             # Maximum value of heart expansion
            max_opening_frame = self.opening.index(maximum_opening)                         # Index of the frame of the maximum value of heart expansion
            max_opening_frame = self.frames_list[max_opening_frame]                         # Frame of the maximum value of heart expansion
            minimum_opening = min(self.opening)                                             # Minimum value of heart expansion
            min_opening_frame = self.opening.index(minimum_opening)                         # Index of the frame of the minimum value of heart expansion 
            min_opening_frame = self.frames_list[min_opening_frame]                         # Frame of the minimum value of heart expansion
            mean = round(sum(self.opening)/len(self.opening),2)                             # Mean of the heart expansions
            var = round((sum([n**2 for n in self.opening])/len(self.opening)),2) - mean     # Variance of the heart expansions
            stand_deviation= round(math.sqrt(var),2)                                        # Standar deviation of the heart expansions
            
            # Making sure if the directory exists, if it doesn't a new directory is creates
            if (os.path.isdir(self.video_path[:-4]) == False):
                os.mkdir(self.video_path[:-4])
                
            # The excel file is created in the previous directory 
            workbook = Workbook(self.video_path[:-4] + '/Statistical results.xlsx')
            Report_Sheet = workbook.add_worksheet()
            
            # The column headers are written.
            Report_Sheet.write(0, 0, 'Frame number')
            Report_Sheet.write(0, 1, 'Heart expansion (Pixels)')
            Report_Sheet.write(0,3,'Frame number max expansion')
            Report_Sheet.write(0,4,'Max heart expansion (Pixels)')
            Report_Sheet.write(0,5,'Frame number min expansion')
            Report_Sheet.write(0,6,'Min heart expansion (Pixels)')
            Report_Sheet.write(0,7,'Mean (Pixels)')
            Report_Sheet.write(0,8,'Variance (Pixels^2)')
            Report_Sheet.write(0,9,'Standard deviation (Pixels)')
        
            
            # The column data are written.
            Report_Sheet.write_column(1, 0, self.frames_list)
            Report_Sheet.write_column(1, 1, self.opening)
            Report_Sheet.write_column(1,3,[max_opening_frame])
            Report_Sheet.write_column(1,4,[maximum_opening])
            Report_Sheet.write_column(1,5,[min_opening_frame])
            Report_Sheet.write_column(1,6,[minimum_opening])
            Report_Sheet.write_column(1,7,[mean])
            Report_Sheet.write_column(1,8,[var])
            Report_Sheet.write_column(1,9,[stand_deviation])
            
            # Finally, the file is closed and the graph removed
            os.remove('graph.tif')
            workbook.close()
        except:
            pass

# Main implementation      
if __name__ == "__main__":
    
    dirname = os.path.dirname(PyQt5.__file__)
    plugin_path = os.path.join(dirname, 'plugins', 'platforms')
    QtWidgets.QApplication.addLibraryPath(plugin_path)
    app =  QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    quit(app.exec_())
    

