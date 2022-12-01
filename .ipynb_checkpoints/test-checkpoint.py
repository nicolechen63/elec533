import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import tflite_runtime.interpreter as tflite

class Camera:

    height = 240
    width = 320
    model_size = 320
    fps = 10
    red_threshold = 0.2 * height/2 * width
    conf_threshold = 0.5
    iou_threshold = 0.45
    stop_threshold = 0.2 * height/2 * width
    data = dict()

    def __init__(self):
        self.cam = cv2.VideoCapture(2)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cam.set(cv2.CAP_PROP_FPS, self.fps)
        self.data["red_mask"] = []
        self.data["k"] = []
        self.data["time"] = []
        self.interpreter = tflite.Interpreter(
            model_path="/home/debian/elec533/best-fp16-2.tflite", num_threads=2)
        self.interpreter.allocate_tensors()
        self.output = self.interpreter.get_output_details()[0]
        self.input = self.interpreter.get_input_details()[0]
        self.data["stop"] = []

    def detect_stop(self, frame):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.model_size, self.model_size))
        frame = frame.astype("float32")/255
        input_data = np.expand_dims(frame, axis=0)
        # scale, zero_point = self.input['quantization']
        # input_data = (input_data / scale + zero_point).astype(np.uint8)

        self.interpreter.set_tensor(self.input['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output['index'])[0]

        # scale, zero_point = self.output['quantization']
        class_id = []
        boxes = []
        confs = []
        for i in range(output_data.shape[0]):
            # confidence = (output_data[i][4] - zero_point) * scale
            confidence = output_data[i][4]
            if confidence > self.conf_threshold:
                # center_x = int((output_data[i][0] - zero_point) * scale * self.width)
                # center_y = int((output_data[i][1] - zero_point) * scale * self.height)
                # width = int((output_data[i][2] - zero_point) * scale * self.width)
                # height = int((output_data[i][3] - zero_point) * scale * self.height)
                center_x = int(output_data[i][0] * self.width)
                center_y = int(output_data[i][1] * self.height)
                width = int(output_data[i][2] * self.width)
                height = int(output_data[i][3] * self.height)
                left = center_x - width / 2
                top = center_y - height / 2
                class_id.append(0)
                confs.append(float(confidence))
                boxes.append([left, top, width, height])
        indices = cv2.dnn.NMSBoxes(
            boxes, confs, self.conf_threshold, self.iou_threshold)
        self.display_box(boxes, confs, indices)
        for index in indices:
            i = index[0]
            box = boxes[i]
            box_size = box[2]*box[3]
            if box_size > self.stop_threshold:
                return True
        return False

    def detect_red(self, hsv):
        '''
        detect_red color with a given threshold. <:
        '''
        lower_red = np.array([140, 60, 40], dtype="uint8")
        upper_red = np.array([200, 255, 255], dtype="uint8")
        mask = cv2.inRange(hsv, lower_red, upper_red)
        # cv2.imshow("red",mask)
        count = mask.sum()/255
        self.data["red_mask"].append(count)
        return (count > self.red_threshold)

    def detect_blue(self, hsv):
        '''
        detect_red color with a given threshold. <:
        '''
        lower_blue = np.array([85, 115, 0], dtype="uint8")
        upper_blue = np.array([155, 255, 255], dtype="uint8")
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # cv2.imshow("blue",mask)
        return mask

    def display_box(self, boxes, confs, indices, line_color=(0, 255, 0), line_width=1):
        # this just displays the boundary lines we previously found on the image
        line_image = np.zeros([self.height, self.width, 3], np.uint8)

        for index in indices:
            i = index[0]
            left = int(boxes[i][0])
            top = int(boxes[i][1])
            width = int(boxes[i][2])
            height = int(boxes[i][3])
            # line displayed here
            cv2.rectangle(line_image, (left, top), (left+width, top+height), line_color, line_width)
            cv2.putText(line_image, str(confs[i]), (left, top+height), cv2.FONT_HERSHEY_SIMPLEX, 1, line_color, line_width)
        cv2.imshow("stop", line_image)

    def display_lines(self, lines, line_color=(0, 255, 0), line_width=1):
        # this just displays the boundary lines we previously found on the image
        line_image = np.zeros([self.height, self.width, 3], np.uint8)

        for line in lines:
            for x1, y1, x2, y2 in line:
                # line displayed here
                cv2.line(line_image, (x1, y1), (x2, y2),
                         line_color, line_width)
        cv2.imshow("lines", line_image)

    def detect_lines(self, mask):
        edges = cv2.Canny(mask, 50, 100)
        cv2.imshow("edges",edges)
        rho = 1
        theta = np.pi / 180
        min_threshold = 40
        line_segments = cv2.HoughLinesP(edges, rho, theta, min_threshold,
                                        np.array([]), minLineLength=20, maxLineGap=20)
        if line_segments is None:
            return 0
        # self.display_lines(line_segments)
        left = []
        right = []
        for line in line_segments:
            theta = math.atan((line[0][3]-line[0][1])/(line[0][2]-line[0][0]+0.1))
            if theta > 0.1:
                right.append(theta)
            elif theta < -0.1:
                left.append(theta)
        left_k = np.average(left) if left else 0
        right_k = np.average(right) if right else 0
        return (left_k + right_k)

    def update(self):
        _, frame = self.cam.read()
        cv2.imshow("frame",frame)
        self.data["time"].append(time.time())
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red = self.detect_red(hsv[int(self.height/2):, :, :])
        blue_mask = self.detect_blue(hsv)
        k = self.detect_lines(blue_mask)
        stop = self.detect_stop(frame)
        return [red, stop, k]

    def close(self):
        self.cam.release()

camera = Camera()
start_time = time.time()
while (True):
    print(time.time()-start_time)
    start_time = time.time()
    camera.update()
    if cv2.waitKey(1) & 0xFF == ord('q'): # press exit
        camera.close()
        break