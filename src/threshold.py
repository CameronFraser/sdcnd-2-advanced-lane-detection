import cv2
import numpy as np


class Threshold:
    def s_channel(self, channel, threshold, binary):
        s_binary = np.zeros_like(channel)
        s_binary[(channel >= threshold[0]) & (channel <= threshold[1])] = 1

        combined_binary = np.zeros_like(binary)
        combined_binary[(s_binary == 1) | (binary == 1)] = 1
        return combined_binary

    def gradient(self, img, orientation, sobel_kernel, sobel_threshold, s_channel_threshold):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
    
        if orientation == 'x':
            sobel = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        else:
            sobel = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1)
        
        abs_sobel = np.abs(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= sobel_threshold[0]) & (scaled_sobel <= sobel_threshold[1])] = 1
        
        return self.s_channel(s_channel, s_channel_threshold, sobel_binary)
    
    def magnitude(self, img, sobel_kernel, threshold, s_channel_threshold):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        scale_factor = np.max(gradient_magnitude) / 255
        gradient_magnitude = (gradient_magnitude/scale_factor).astype(np.uint8)

        binary_output = np.zeros_like(gradient_magnitude)
        binary_output[(gradient_magnitude >= threshold[0]) & (gradient_magnitude <=threshold[1])] = 1
        return self.s_channel(s_channel, s_channel_threshold, binary_output)
    
    def direction(self, img, sobel_kernel, threshold, s_channel_threshold):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        abs_gradient_direction = np.arctan2(np.abs(sobel_y), np.abs(sobel_x))

        binary_output = np.zeros_like(abs_gradient_direction)
        binary_output[(abs_gradient_direction >= threshold[0]) & (abs_gradient_direction <= threshold[1])] = 1
        return self.s_channel(s_channel, s_channel_threshold, binary_output)

    def combine(self, grad_x, grad_y, magnitude, direction):
        combined = np.zeros_like(direction)
        combined[((grad_x == 1) & (grad_y == 1)) | ((magnitude == 1) & (direction == 1))] = 1
        return combined
        
