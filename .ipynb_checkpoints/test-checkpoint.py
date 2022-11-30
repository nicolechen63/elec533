import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class Camera:

    height = 120
    width = 160
    fps = 10
    red_threshold = 0.2 * height/2 * width
    data = dict()

    def init(self):
        self.cam = cv2.VideoCapture(2)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cam.set(cv2.CAP_PROP_FPS, self.fps)
        self.data["red_mask"] = []
        self.data["k"] = []
        self.data["time"] = []

    def detect_red(self,hsv):
        '''
        detect_red color with a given threshold. <:
        '''
        lower_red = np.array([150, 70, 50], dtype="uint8")
        upper_red = np.array([180, 255, 255], dtype="uint8")
        mask = cv2.inRange(hsv, lower_red, upper_red)
        cv2.imshow("red",mask)
        count = mask.sum()/255
        self.data["red_mask"].append(count)
        return (count>self.red_threshold)

    def detect_blue(self,hsv):
        '''
        detect_red color with a given threshold. <:
        '''
        lower_blue = np.array([90, 120, 0], dtype = "uint8")
        upper_blue = np.array([150, 255, 255], dtype="uint8")
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        cv2.imshow("blue",mask)
        return mask

    def display_lines(self, lines, line_color=(0, 255, 0), line_width=1):
        #this just displays the boundary lines we previously found on the image
        line_image = np.zeros([int(self.height/2), self.width,3], np.uint8)

        for line in lines:
            for x1, y1, x2, y2 in line:
                #line displayed here
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
        cv2.imshow("lines",line_image)

    def detect_lines(self, mask):
        edges = cv2.Canny(mask, 50, 100)
        cv2.imshow("edges",edges)
        rho = 1
        theta = np.pi / 180
        min_threshold = 20
        line_segments = cv2.HoughLinesP(edges, rho, theta, min_threshold,
                                        np.array([]), minLineLength=10, maxLineGap=10)
        if line_segments is None:
            return 0
        self.display_lines(line_segments)
        left = []
        right = []
        for line in line_segments:
            k = (line[0][3]-line[0][1])/(line[0][2]-line[0][0] + 0.1)
            if k > 0.1:
                right.append(k)
            elif  k < -0.1:
                left.append(k)
        left_k = np.average(left) if left else 0
        right_k = np.average(right) if right else 0
        return (left_k + right_k)
    
    def update(self):
        _, frame = self.cam.read()
        cv2.imshow("frame",frame)
        self.data["time"].append(time.time())
        hsv = cv2.cvtColor(frame[int(self.height/2):,:,:], cv2.COLOR_BGR2HSV)
        red = self.detect_red(hsv)
        blue_mask = self.detect_blue(hsv)
        k = self.detect_lines(blue_mask)
        return [red, k]

    def close(self):
        self.cam.release()        

camera = Camera()
camera.init()
while (True):
    camera.update()
    if cv2.waitKey(1) & 0xFF == ord('q'): # press exit
        camera.close()
        break
