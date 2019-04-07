import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from camera import Camera
from plot import plot_images
from image import load_images
from pipeline import *

def pipeline_setup(calibration_images, pattern_size, draw=False):
    camera = Camera()
    chessboard_images = camera.calibrate(calibration_images, pattern_size)
    if draw == True:
        plot_images(chessboard_images, 4)
    
    return camera

def pipeline_draw_ui(undist_img, unwarped_img, left_curvature, right_curvature, offset):
    result = cv2.addWeighted(undist_img, 1, unwarped_img, 0.3, 0)
    
    result_like = np.zeros_like(unwarped_img)
    cv2.rectangle(result_like, (20, 20), (result_like.shape[1] - 20, 200), [30, 30, 30], -1)
    
    left_curve_text = "Radius of left curve: %.2f m" % left_curvature
    cv2.putText(result_like, left_curve_text, (100, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), thickness=2)
    
    right_curve_text = "Radius of right curve: %.2f m" % right_curvature
    cv2.putText(result_like, right_curve_text, (750, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), thickness=2)
    
    offset_text = "Offset of vehicle: %.2f m" % offset
    cv2.putText(result_like, offset_text, (int(result_like.shape[1] / 2 - 180), 140), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), thickness=2)
    return cv2.addWeighted(result, 0.8, result_like, 1., 0.)

def pipeline_process(camera, draw=False):
    lane = None
    gradx_params = [3, (30, 100), (170, 230)] # kernel size, sobel threshold, s channel threshold
    grady_params = [3, (40, 100), (170, 230)] # kernel size, sobel threshold, s channel threshold
    mag_params = [11, (40, 100), (170, 230)] # kernel size, sobel threshold, s channel threshold
    dir_params = [15, (0.8, 1.2), (170, 230)] # kernel size, sobel threshold, s channel threshold
    nwindows = 9
    margin = 100
    minpix = 25

    def process(img):
        nonlocal lane
        undist_img = camera.undistort_image(img)
        threshold_img = detect_edges(undist_img, gradx_params, grady_params, mag_params, dir_params)
        warped_img, _, M_inv = transform_perspective(threshold_img, camera)
        lane = find_lane_lines(lane, warped_img, nwindows, margin, minpix)
        left_curvature, right_curvature, offset = lane.measure_curvature_and_offset()
        color_warp = lane.fill_lane()
        unwarped_img = cv2.warpPerspective(color_warp, M_inv, (img.shape[1], img.shape[0]))
        return pipeline_draw_ui(undist_img, unwarped_img, left_curvature, right_curvature, offset) 
    return process


file_root = os.path.dirname(os.path.abspath(__file__))
calibration_images = load_images(glob.glob(os.path.join(file_root, '../camera_cal/calibration*.jpg')))
test_image_paths = glob.glob(os.path.join(file_root, '../test_images/*.jpg'))

test_images = load_images(test_image_paths)

def process_images(calibration_images, test_images, file_root):
    camera = pipeline_setup(calibration_images, (9, 6))

    for i, image in enumerate(test_images):
        process_fn = pipeline_process(camera)
        result = process_fn(image)
        filename = test_image_paths[i].split('/')[-1]
        filename = filename.split('.')[0] + '.png'
        plt.imsave(os.path.join(file_root, '../output_images/', filename), result)


def process_video(file_path, file_path_output):
    camera = pipeline_setup(calibration_images, (9, 6))

    clip1 = VideoFileClip(file_path)

    process_fn = pipeline_process(camera)
    clip = clip1.fl_image(process_fn)
    clip.write_videofile(file_path_output, audio=False)


process_images(calibration_images, test_images, file_root)

process_video(os.path.join(file_root, '../project_video.mp4'), os.path.join(file_root, '../test_videos_output/project_video.mp4'))
process_video(os.path.join(file_root, '../challenge_video.mp4'), os.path.join(file_root, '../test_videos_output/challenge_video.mp4'))
process_video(os.path.join(file_root, '../harder_challenge_video.mp4'), os.path.join(file_root, '../test_videos_output/harder_challenge_video.mp4'))



