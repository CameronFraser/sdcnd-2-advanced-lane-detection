import cv2
import numpy as np

class Camera:
    def __init__(self):
        self.obj_points = []
        self.img_points = []
        self.calibration_images = []
        self.mtx = []
        self.dist = []
        self.rvecs = []
        self.tvecs = []
    
    def calibrate(self, images, pattern_size):
        drawn_images = []
        self.calibration_images = images
        self.pattern_size = pattern_size
        objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        
        for img in self.calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
            if ret == True:
                self.img_points.append(corners)
                self.obj_points.append(objp)
                drawn_images.append(cv2.drawChessboardCorners(img, self.pattern_size, corners, ret))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, img.shape[1::-1], None, None)
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        return drawn_images

    def undistort_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    
    def undistort_images(self, images):
        undistorted_images = []
        for img in images:
            undistorted_images.append(self.undistort_image(img))
        return undistorted_images

    def transform_perspective(self, img, src, dst):
        M = cv2.getPerspectiveTransform(src, dst)
        try:
            height, width = img.shape
        except:
            height, width, _ = img.shape

        warped = cv2.warpPerspective(img, M, (width, height))

        return warped, M