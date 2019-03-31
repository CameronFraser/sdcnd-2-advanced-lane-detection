from camera import Camera
import os
import glob

file_root = os.path.dirname(os.path.abspath(__file__))

calibration_images = glob.glob(os.path.join(file_root, '../camera_cal/calibration*.jpg'))
pattern_size = (8, 6)

camera = Camera()
camera.calibrate(calibration_images, pattern_size)

