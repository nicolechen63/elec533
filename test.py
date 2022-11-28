
import cv2

# camera resolution setting
CAM_H = 480
CAM_W = 640 

cam = cv2.VideoCapture(2)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)

while (True):
    _, frame = cam.read()

    cv2.imshow('img', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # press exit
        break
