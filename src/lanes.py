import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

class Lane:
    def __init__(self, img):
        self.img = img
    
    def find_pixels(self, nwindows = 9, margin = 100, minpix = 50):
        height = self.img.shape[0]
        histogram = np.sum(self.img[height // 2:,:], axis=0)
        out_img = np.dstack((self.img, self.img, self.img))

        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = np.int(height // nwindows)

        nonzero = self.img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_indices = []
        right_lane_indices = []

        for window in range(nwindows):
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 5)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 5)

            good_left_indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_indices.append(good_left_indices)
            right_lane_indices.append(good_right_indices)

            if len(good_left_indices) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_indices]))
            if len(good_right_indices) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_indices]))
        print(out_img[win_xleft_low + 2][win_y_low - 1])
        try:
            left_lane_indices = np.concatenate(left_lane_indices)
            right_lane_indices = np.concatenate(right_lane_indices)
        except ValueError:
            pass
        
        leftx = nonzerox[left_lane_indices]
        lefty = nonzeroy[left_lane_indices]
        rightx = nonzerox[right_lane_indices]
        righty = nonzeroy[right_lane_indices]

        self.leftx = leftx
        self.rightx = rightx
        self.lefty = lefty
        self.righty = righty
        self.out_img = out_img

        return leftx, lefty, rightx, righty, out_img
    
    def fit_polynomial(self):
        height = self.img.shape[0]
        left_fit = np.polyfit(self.lefty, self.leftx, 2)
        right_fit = np.polyfit(self.righty, self.rightx, 2)

        ploty = np.linspace(0, height - 1, height)

        try:
            left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        except TypeError:
            left_fitx = 1 * ploty**2 + 1 * ploty
            right_fitx = 1 * ploty**2 + 1 * ploty
        
        self.out_img[self.lefty, self.leftx] = [255, 0, 0]
        self.out_img[self.righty, self.rightx] = [0, 0, 255]

        #plt.plot(left_fitx, ploty, color="yellow")
        #plt.plot(right_fitx, ploty, color="yellow")

        return self.out_img