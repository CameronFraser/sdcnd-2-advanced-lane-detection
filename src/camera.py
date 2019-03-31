import cv2
import numpy as np
import matplotlib.image as mpimg
from plot import plot_images

class Camera:
    def __init__(self):
        self.obj_points = []
        self.img_points = []
        self.calibration_images = []
        self.drawn_images = []
    
    def calibrate(self, images, pattern_size):
        self.calibration_images = images
        self.pattern_size = pattern_size
        objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        
        for file_name in self.calibration_images:
            ret, corners, img = self.find_corners(file_name)
            if ret == True:
                self.img_points.append(corners)
                self.obj_points.append(objp)
                self.drawn_images.append(cv2.drawChessboardCorners(img, self.pattern_size, corners, ret))
    
    def find_corners(self, file_name):
        img = mpimg.imread(file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        return ret, corners, img

        
    def show_chessboards(self):
        plot_images(self.drawn_images)