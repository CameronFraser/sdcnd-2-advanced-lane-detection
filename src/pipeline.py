import numpy as np
import cv2

from edgedetection import EdgeDetection
from schannel import s_channel_threshold
from lanes import Lane
from plot import plot_images


def detect_edges(img, gradx_params, grady_params, mag_params, dir_params, plot_steps=False):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    edge_d = EdgeDetection()

    grad_x_binary = edge_d.gradient(img, 'x', gradx_params[0], gradx_params[1], l_channel)
    grad_x_binary_s = s_channel_threshold(s_channel, gradx_params[2], grad_x_binary)

    grad_y_binary = edge_d.gradient(img, 'y', grady_params[0], grady_params[1], l_channel)
    grad_y_binary_s = s_channel_threshold(s_channel, grady_params[2], grad_y_binary)

    mag_binary = edge_d.magnitude(img, mag_params[0], mag_params[1], l_channel)
    mag_binary_s  = s_channel_threshold(s_channel, mag_params[2], mag_binary)
    
    dir_binary = edge_d.direction(img, dir_params[0], dir_params[1], l_channel)
    dir_binary_s = s_channel_threshold(s_channel, dir_params[2], dir_binary)

    combined = edge_d.combine(grad_x_binary_s, grad_y_binary_s, mag_binary_s, dir_binary_s)
    
    if plot_steps == True:
        threshold_images = []
        threshold_images.append(grad_x_binary)
        threshold_images.append(grad_y_binary)
        threshold_images.append(mag_binary)
        threshold_images.append(dir_binary)
        threshold_images.append(grad_x_binary_s)
        threshold_images.append(grad_y_binary_s)
        threshold_images.append(mag_binary_s)
        threshold_images.append(dir_binary_s)
        plot_images(threshold_images, 2)
    
    return combined

def transform_perspective(image, camera):
    try:
        height, width = image.shape
    except:
        height, width, _ = image.shape

    offset = 150
    upper_left_src = [width/2 - 50, height/2 + 100]
    upper_right_src = [width/2 + 50, height/2 + 100]
    bottom_left_src = [0+offset, height - 20]
    bottom_right_src = [width-offset, height - 20]

    src = np.float32([upper_left_src, upper_right_src, bottom_right_src, bottom_left_src])

    lines = np.array(src, np.int32)

    upper_left_dst = [0+offset*2+10, 0]
    upper_right_dst = [width-offset*2, 0]
    bottom_left_dst = [0+offset+20, height]
    bottom_right_dst = [width-offset-20, height]

    dst = np.float32([upper_left_dst, upper_right_dst, bottom_right_dst, bottom_left_dst])
    
    return camera.transform_perspective(image, src, dst)

def find_lane_lines(lane, img, nwindows, margin, minpix):
    if lane == None:
        lane = Lane(img)
        lane.find_pixels(nwindows, margin, minpix)
    else:
        left_fit = lane.left_fit
        right_fit = lane.right_fit
        lane.img = img
        lane.search_around_poly(left_fit, right_fit)
    lane.fit_polynomial()
    return lane
