import cv2
import numpy as np
import math
import sys
import time
# Import packages
import os
import argparse
from threading import Thread
import importlib.util
#import matplotlib.pyplot as plt
#import Adafruit_BBIO.PWM as PWM


def detect_red(frame,threshold=1000):
    '''
    detect_red color with a given threshold. <:
    '''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([150, 70, 50], dtype="uint8")
    upper_red = np.array([179, 255, 255], dtype="uint8")
    mask = cv2.inRange(hsv, lower_red, upper_red)
    #cv2.imshow("Red_color",mask)
    return mask.sum()>threshold

def detect_edges(frame):
    #print("Start ED")
    # filter for blue lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.imshow("HSV",hsv)
    lower_blue = np.array([90, 120, 0], dtype = "uint8")
    upper_blue = np.array([150, 255, 255], dtype="uint8")
    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    #cv2.imshow("mask",mask)

    # detect edges
    edges = cv2.Canny(mask, 50, 100)
    #cv2.imshow("edges",edges)
    #print("End ED")
    return edges

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus lower half of the screen
    polygon = np.array([[
        (0, height),
        (0,  height/2),
        (width , height/2),
        (width , height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)

    cropped_edges = cv2.bitwise_and(edges, mask)
    #cv2.imshow("roi",cropped_edges)

    return cropped_edges

def detect_line_segments(cropped_edges):
    #detects line segment via HoughLines
    rho = 1
    theta = np.pi / 180
    min_threshold = 10

    line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold,
                                    np.array([]), minLineLength=5, maxLineGap=150)

    return line_segments

def average_slope_intercept(frame, line_segments):
    #finds average slope intercept
    lane_lines = []

    if line_segments is None:
        #print("no line segments detected")
        return lane_lines

    height, width,_ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2: #skip if slope is infinity
                continue

            #find line
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)

            #plot to where it converges at the center
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    return lane_lines

def make_points(frame, line):
    #helper function for drawing
    height, width, _ = frame.shape

    slope, intercept = line

    y1 = height  # bottom of the frame
    y2 = int(y1 / 2)  # make points from middle of the frame down

    if slope == 0:
        slope = 0.1

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return [[x1, y1, x2, y2]]

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=6):
    #this just displays the boundary lines we previously found on the image
    line_image = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                #line displayed here
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)

    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

def get_steering_angle(frame, lane_lines):
    #produces the steering angle for the heading line and 
    #used to find deviation
    height,width,_ = frame.shape

    if len(lane_lines) == 2:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2)

    elif len(lane_lines) == 1:
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        y_offset = int(height / 2)

    elif len(lane_lines) == 0:
        x_offset = 0
        y_offset = int(height / 2)

    #fun cosine math to find angle
    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90

    return steering_angle
    
def cam_test(res):
    '''
    Helper function for testing.
    Seeing the camera input and lines being detected.
    '''
    #Capturing
    video = cv2.VideoCapture(2)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #320
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #240
    while(True):      
        ret,frame = video.read()
        #print("ret:", ret)
        #print("frame shape:", frame.shape)
        frame = cv2.resize(frame, (res, res))
        #print("res:",res)
        #print("frame shape:", frame.shape)
        #frame = cv2.flip(frame,-1)
        cv2.imshow("original",frame)
        edges = detect_edges(frame)
        roi = region_of_interest(edges)
        line_segments = detect_line_segments(roi)
        lane_lines = average_slope_intercept(frame,line_segments)
        lane_lines_image = display_lines(frame,lane_lines)
        cv2.imshow("lane_lines_image",lane_lines_image)
        steering_angle = get_steering_angle(frame, lane_lines)
        heading_image = display_heading_line(lane_lines_image,steering_angle)
        cv2.imshow("heading line",heading_image)
        print("Frame processed\n")
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
           break
    video.release()
    cv2.destroyAllWindows()
    
def main(res=100, kpr=12, kdr=8, driving_speed = 7.92 ):
    steering_port = "P9_14"
    driving_port = "P8_13"
    
    # Set up PWM
    #PWM.start(steering_port, 7.5, 50)
    #PWM.start(driving_port, 7.5, 50)
    
    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                        required=True)
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.5)
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='1280x720')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')
    
    args = parser.parse_args()
    
    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = float(args.threshold)
    resW, resH = args.resolution.split('x')
    imW, imH = int(resW), int(resH)
    use_TPU = args.edgetpu
    
    # Import TensorFlow libraries
    # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate
    
    # If using Edge TPU, assign filename for Edge TPU model
    if use_TPU:
        # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'       
    
    # Get path to current working directory
    CWD_PATH = os.getcwd()
    
    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
    
    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)
    
    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    
    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])
    
    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        print(PATH_TO_CKPT)
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)
    
    interpreter.allocate_tensors()
    
    
        # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    
    floating_model = (input_details[0]['dtype'] == np.float32)
    
    input_mean = 127.5
    input_std = 127.5
    
    # Check output layer name to determine if this model was created with TF2 or TF1,
    # because outputs are ordered differently for TF2 and TF1 models
    outname = output_details[0]['name']
    #print("outname:", outname)
    #print("ouput details:", output_details)
    
    if ('StatefulPartitionedCall' in outname): # This is a TF2 model
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else: # This is a TF1 model
        boxes_idx, classes_idx, scores_idx = 0, 1, 2
        
    print("boxes_idx, classes_idx, scores_idx: ", boxes_idx, classes_idx, scores_idx)    
    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    
    # Initialize video stream
    #videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    #time.sleep(1)

    #Capturing
    video = cv2.VideoCapture(2)

    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #320
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #240

    #Saving Vid
    savedvid = cv2.VideoWriter('carvid.avi',cv2.VideoWriter.fourcc(*'MJPG'), 10 , (res,res))

    # Calibration Setup
    rotbase = 7.5
    lastTime = 0
    lastError = 0
    j= 0

    #Adding data points
    der_r = np.array([])
    pro_r = np.array([])
    err = np.array([])
    steering = np.array([])
    speed = np.array([])
    
    time.sleep(3) # calibration for esc
    
    #Red detection
    red_detected = False
    a=50
    
    while(True):
        ret,frame = video.read()
        print("ret:", ret)
        print("frame shape:", frame.shape)
        frame = cv2.resize(frame, (res, res))
        print("res:", res)
        print("frame shape:", frame.shape)
        
        #Don't attempt to read the stop sign for the first a's iteration.
        if (j>a):
            if detect_red(frame):
                if red_detected: #Terminate if we already saw stop sign before.
                  break
                a = j + 100
                red_detected = True
                #print("red detected so pwm set duty cycle")
                #PWM.set_duty_cycle(driving_port,7.5)
                time.sleep(2)
        
        # Acquire frame and resize to expected shape [1xHxWx3]
        frame1 = frame.copy()
        frame_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
    
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
    
        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()
        
        #predicted_id = np.zeros(batch_size)
    
        # Retrieve detection results
        #print("outputdetails : ", output_details)
        #print("outputdetails : ", type(output_details))
        output_data = interpreter.get_tensor(output_details[0]['index'])
        #print("output_data:", output_data)
        #print("type of outputdata: ", type(output_data))
        #predicted_id[i] = np.argmax(output_data)
        #print("boxes idx:", boxes_idx)
        #print(interpreter.get_tensor(output_details[0]['index'])[0])
        
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        print("boxes:")
        #cv2.imshow('Object detector box: ', boxes)
        
        #print(output_details[boxes_idx]['index'])[0]
        
        #boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects  #classes_idx
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects  scores_idx
        
        #Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
        
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        
                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

        
        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)
        
        #Edge Detection
        #edges = detect_edges(frame)
        #roi = region_of_interest(edges)
        #line_segments = detect_line_segments(roi)
        #lane_lines = average_slope_intercept(frame,line_segments)
        #steering_angle = get_steering_angle(frame, lane_lines)
        #print("steering angle", steering_angle)
        
        #Showing and Saving Video - Comment Out for better performance
        #lane_lines_image = display_lines(frame,lane_lines) 
        #savedvid.write(lane_lines_image) #heading_image
        ##cv2.imshow("heading line",heading_image)
        #cv2.imshow("heading line",lane_lines_image)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        j += 1

    savedvid.release()
    video.release()
    cv2.destroyAllWindows()
    

#cam_test(res=80) #For testing the camera
main(res=250, kpr=10, kdr=0, driving_speed =7.88)