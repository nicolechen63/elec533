# %%
# ref: https://www.hackster.io/rlk3/young-boys-0b1d78
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from IPython.display import clear_output
import tflite_runtime.interpreter as tflite


# %%
class PidClass:
    p = 0
    i = 0
    d = 0

    error_last = 0
    error_last2 = 0

    def __init__(self, p, i, d):
        self.p = p
        self.i = i
        self.d = d

    def get(self, error):
        # p
        output = self.p * (error - self.error_last)
        # i
        output += self.i * error
        # d
        output += self.d * (error - 2*self.error_last + self.error_last2)
        self.error_last2 = self.error_last
        self.error_last = error
        return output


class PidClass2:
    p = 0
    i = 0
    d = 0

    error_i = 0
    error_last = 0

    def __init__(self, p, i, d):
        self.p = p
        self.i = i
        self.d = d

    def get(self, error):
        # p
        output = self.p * error
        # i
        self.error_i += error
        output += self.i * self.error_i
        # d
        output += self.d * (error - self.error_last)
        self.error_last = error
        return output


class Speed:

    target_speed = 0
    current_speed = 0
    current_count = 0
    current_output = 0
    data = dict()

    def __init__(self, target_speed, p, i, d):
        self.target_speed = target_speed
        self.pid = PidClass(p, i, d)
        self.data["current_count"] = []
        self.data["current_speed"] = []
        self.data["current_output"] = []
        self.data["time"] = []

    def get_count(self):
        with open('/sys/module/hello/parameters/count', 'r') as filetowrite:
            new_count = filetowrite.readline()
            new_count = new_count[:-1]
            new_count = int(new_count)
            count = new_count - self.current_count
            self.current_count = new_count
            return count

    def get_time(self, i):
        with open('/sys/module/hello/parameters/x' + str(i), 'r') as filetowrite:
            time = filetowrite.readline()
            time = time[:-1]
            time = int(time)
            return time

    def get_speed(self):
        count = self.get_count()
        self.data["current_count"].append(count)
        if count == 0:
            self.data["current_speed"].append(0)
            return 0
        count = min(count, 5)
        time_list = np.zeros(count)
        for i in range(count):
            time_list[i] = (self.get_time(i)+1)
        speed = 1000/np.average(time_list)
        self.data["current_speed"].append(speed)
        return speed

    def init_pwm(self):
        with open('/dev/bone/pwm/1/a/period', 'w') as filetowrite:
            filetowrite.write('20000000')
        with open('/dev/bone/pwm/1/a/duty_cycle', 'w') as filetowrite:
            filetowrite.write('1550000')
        with open('/dev/bone/pwm/1/a/enable', 'w') as filetowrite:
            filetowrite.write('1')

    def set_speed(self, value):
        value = max(0, min(100, value))
        self.current_output = value
        pwd = str(int(value * 4000 + 1550000))
        with open('/dev/bone/pwm/1/a/duty_cycle', 'w') as filetowrite:
            filetowrite.write(pwd)
        return value

    def update(self):
        self.data["time"].append(time.time())
        self.current_speed = self.get_speed()
        speed_error = self.target_speed - self.current_speed
        speed_pid = self.pid.get(speed_error)
        self.current_output += speed_pid
        self.current_output = self.set_speed(self.current_output)
        self.data["current_output"].append(self.current_output)


class Camera:

    height = 240
    width = 320
    model_size = 320
    fps = 10
    red_threshold = 0.7 * height/2 * width
    conf_threshold = 0.8
    iou_threshold = 0.45
    #stop_threshold = 0.2 * height * width
    data = dict()

    def __init__(self):
        self.cam = cv2.VideoCapture(2)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cam.set(cv2.CAP_PROP_FPS, self.fps)
        self.data["red_mask"] = []
        self.data["k"] = []
        self.data["time"] = []
        self.data["frame"] = []
        self.interpreter = tflite.Interpreter(
            model_path="/home/debian/elec533/best-fp16.tflite", num_threads=2)
        self.interpreter.allocate_tensors()
        self.output = self.interpreter.get_output_details()[0]
        self.input = self.interpreter.get_input_details()[0]
        self.data["stop"] = []

    def detect_stop(self, frame):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.model_size, self.model_size))
        frame = frame.astype(np.float32)/255
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
        # self.display_box(boxes, confs, indices)
        for index in indices:
            i = index[0]
            box = boxes[i]
            box_size = box[2]*box[3]
            #if box_size > self.stop_threshold:
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
        # cv2.imshow("edges",edges)
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
        _, jpg = cv2.imencode(".jpg", frame)
        self.data["frame"].append(jpg)
        # cv2.imshow("frame",frame)
        self.data["time"].append(time.time())
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red = self.detect_red(hsv[int(self.height/2):, :, :])
        blue_mask = self.detect_blue(hsv)
        k = self.detect_lines(blue_mask)
        stop = self.detect_stop(frame)
        return [red, stop, k]

    def close(self):
        self.cam.release()
        for i,jpg in enumerate(self.data["frame"]):
            with open(str(i)+'.jpg', 'wb') as filetowrite:
                filetowrite.write(jpg.tobytes())



class Direction:

    target_direction = 0
    current_direction = 0
    current_output = 0
    data = dict()

    def __init__(self, p, i, d):
        self.pid = PidClass2(p, i, d)
        self.data["current_direction"] = []
        self.data["current_output"] = []
        self.data["red"] = []
        self.data["time"] = []
        self.cam = Camera()

    def init_pwm(self):
        with open('/dev/bone/pwm/1/b/period', 'w') as filetowrite:
            filetowrite.write('20000000')
        with open('/dev/bone/pwm/1/b/duty_cycle', 'w') as filetowrite:
            filetowrite.write('1500000')
        with open('/dev/bone/pwm/1/b/enable', 'w') as filetowrite:
            filetowrite.write('1')

    def set_direction(self, value):
        value = max(-100, min(100, value))
        pwd = str(int(value * 3000 + 1500000))
        with open('/dev/bone/pwm/1/b/duty_cycle', 'w') as filetowrite:
            filetowrite.write(pwd)
        return value

    def update(self):
        red, stop, k = self.cam.update()
        self.data["time"].append(time.time())
        self.data["current_direction"].append(k)
        direction_pid = self.pid.get(k)
        self.current_output += direction_pid
        self.current_output = self.set_direction(self.current_output)
        self.data["current_output"].append(self.current_output)
        return red, stop

    def close(self):
        self.cam.close()


# %%
#direction.close()


# %%
speed = Speed(200, 0.05, 0.01, 0.01)
speed.init_pwm()
direction = Direction(20, 0, 0)
direction.init_pwm()


# %%
# direction.p = 15

# %%
# plt.plot(direction.data["current_direction"])

# %%
# speed.init_pwm()
# direction.init_pwm()


# %%
# direction.update()

# %%
stop_count = 0
red_cooldown = 30
red_count = 0
sign_count = 0
run_loop = 100
for _ in range(run_loop):
    #if stop_count:
    #    stop_count -= 1
    #    speed.set_speed(0)
    speed.update()
    red, stop = direction.update()
    # if red_cooldown > 0:
    #     red_cooldown -= 1
    # elif red and red_count == 0:
    #     stop_count = 10
    #     red_cooldown = 30
    #     red_count = 1
    # elif red and red_count == 1:
    #     speed.set_speed(0)
    #     break
    # if red_cooldown == 0 and stop and sign_count == 0:
    #     stop_count = 10
    #     sign_count = 1

    # clear_output(wait=True)
    #print(f'Speed: {speed.current_speed}, {speed.current_output}')
speed.set_speed(0)
direction.set_direction(0)


# %%
# # init PWM
# # P9_14 - Speed/ESC
# with open('/dev/bone/pwm/1/a/period', 'w') as filetowrite:
#     filetowrite.write('20000000')
# with open('/dev/bone/pwm/1/a/duty_cycle', 'w') as filetowrite:
#     filetowrite.write('1550000')
# with open('/dev/bone/pwm/1/a/enable', 'w') as filetowrite:
#     filetowrite.write('1')
# # P9_16 - Steering
# with open('/dev/bone/pwm/1/b/period', 'w') as filetowrite:
#     filetowrite.write('20000000')
# with open('/dev/bone/pwm/1/b/duty_cycle', 'w') as filetowrite:
#     filetowrite.write('1500000')
# with open('/dev/bone/pwm/1/b/enable', 'w') as filetowrite:
#     filetowrite.write('1')


# %%
# with open('/dev/bone/pwm/1/b/duty_cycle', 'w') as filetowrite:
#     filetowrite.write('1500000')


# %%
# cam = cv2.VideoCapture(2)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
# cam.set(cv2.CAP_PROP_FPS, 10)
# cam.release()


# %%



