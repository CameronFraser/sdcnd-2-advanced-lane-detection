import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

class Lane:
    def __init__(self, img):
        self.img = img
        self.fit_history = []
    
    def find_pixels(self, nwindows = 9, margin = 100, minpix = 25):
        height = self.img.shape[0]
        histogram = np.sum(self.img[height // 2:,:], axis=0)
        out_img = np.stack((self.img, self.img, self.img), axis=2)

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
        self.margin = margin
        self.out_img = out_img

        return leftx, lefty, rightx, righty, out_img
    
    def average_fits(self):
        left_fitx_sum = 0
        right_fitx_sum = 0
        ploty_sum = 0
        history_len = len(self.fit_history)
        for fit in self.fit_history:
            ploty_sum += fit[0]
            left_fitx_sum += fit[1]
            right_fitx_sum += fit[2]
        
        self.ploty_avg = ploty_sum / history_len
        self.left_fitx_avg = left_fitx_sum / history_len
        self.right_fitx_avg = right_fitx_sum / history_len
             
    
    def enqueue_history(self, fit):
        self.fit_history.append(fit)
        if (len(self.fit_history) >= 20):
            self.fit_history.pop(0)
        
        self.average_fits()
    
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
        
        self.ploty = ploty
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx
        self.left_fit = left_fit
        self.right_fit = right_fit
        self.enqueue_history((ploty, left_fitx, right_fitx))
        return ploty, left_fitx, right_fitx
    
    def fill_lane(self):
        warp_zero = np.zeros_like(self.img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([self.left_fitx_avg, self.ploty_avg]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx_avg, self.ploty_avg])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        return color_warp

    def search_around_poly(self, prev_left_fit, prev_right_fit):
        margin = 200

        nonzero = self.img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        left_lane_indices = ((nonzerox > (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + 
                        prev_left_fit[2] - self.margin)) & (nonzerox < (prev_left_fit[0]*(nonzeroy**2) + 
                        prev_left_fit[1]*nonzeroy + prev_left_fit[2] + self.margin)))
        right_lane_indices = ((nonzerox > (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + 
                        prev_right_fit[2] - self.margin)) & (nonzerox < (prev_right_fit[0]*(nonzeroy**2) + 
                        prev_right_fit[1]*nonzeroy + prev_right_fit[2] +self. margin)))

        leftx = nonzerox[left_lane_indices]
        lefty = nonzeroy[left_lane_indices]
        rightx = nonzerox[right_lane_indices]
        righty = nonzeroy[right_lane_indices]

        self.leftx = leftx
        self.rightx = rightx
        self.lefty = lefty
        self.righty = righty

    def measure_curvature_and_offset(self):
        ym_per_pix = 30/self.img.shape[0]
        xm_per_pix = 3.7/700

        y_eval = np.max(self.ploty_avg)
        left_fit_cr = np.polyfit(self.ploty_avg*ym_per_pix, self.left_fitx_avg*xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.ploty_avg*ym_per_pix, self.right_fitx_avg*xm_per_pix, 2)

        x_max = self.img.shape[1]*xm_per_pix
        y_max = self.img.shape[0]*ym_per_pix
        center = x_max / 2

        left_line = left_fit_cr[0]*y_max**2 + left_fit_cr[1]*y_max + left_fit_cr[2]
        right_line = right_fit_cr[0]*y_max**2 + right_fit_cr[1]*y_max + right_fit_cr[2]
        offset = (left_line + (right_line - left_line)/2) - center

        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
        self.left_curverad = left_curverad
        self.right_curverad = right_curverad
        self.offset = offset
        return left_curverad, right_curverad, offset

