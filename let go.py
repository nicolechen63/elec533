# %%
# ref: User raja_961, “Autonomous Lane-Keeping Car Using Raspberry Pi and OpenCV”. Instructables. URL: https://www.instructables.com/Autonomous-Lane-Keeping-Car-U sing-Raspberry-Pi-and/
# ref: https://www.hackster.io/rlk3/young-boys-0b1d78
# import necessay libraries
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import tflite_runtime.interpreter as tflite


# %%
# incremental PID
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

# location PID


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

# control speed


class Speed:

    target_speed = 0
    current_speed = 0
    current_count = 0
    current_output = 0
    # save the intermidiate value
    data = dict()

    def __init__(self, target_speed, p, i, d):
        self.target_speed = target_speed
        self.pid = PidClass(p, i, d)
        self.data["current_count"] = []
        self.data["current_speed"] = []
        self.data["current_output"] = []
        self.data["time"] = []

    # return the number of counts after last read
    def get_count(self):
        # get the count from the hello module
        with open('/sys/module/hello/parameters/count', 'r') as filetowrite:
            new_count = filetowrite.readline()
            # rm "\n"
            new_count = new_count[:-1]
            # convert str to int
            new_count = int(new_count)
            # how many counts after last read
            count = new_count - self.current_count
            # save current value
            self.current_count = new_count
            return count

    # return the time interal between i'th and (i+1)'th encoder trigers
    def get_time(self, i):
        # parameter: x0 to x4
        with open('/sys/module/hello/parameters/x' + str(i), 'r') as filetowrite:
            time = filetowrite.readline()
            # rm "\n"
            time = time[:-1]
            # convert str to int
            time = int(time)
            return time

    # return current speed
    def get_speed(self):
        # get nunber of counts after last read
        count = self.get_count()
        self.data["current_count"].append(count)
        if count == 0:
            # no movement, return 0
            self.data["current_speed"].append(0)
            return 0
        # we only have the last 5 time intervals
        count = min(count, 5)
        time_list = np.zeros(count)
        for i in range(count):
            # read the time interval
            time_list[i] = (self.get_time(i)+1)
        # average the time interval and calculate the time
        speed = 1000/np.average(time_list)
        self.data["current_speed"].append(speed)
        return speed

    def init_pwm(self):
        # period: 20000000
        with open('/dev/bone/pwm/1/a/period', 'w') as filetowrite:
            filetowrite.write('20000000')
        # duty_cycle: 7.5%
        with open('/dev/bone/pwm/1/a/duty_cycle', 'w') as filetowrite:
            filetowrite.write('1500000')
        # enable
        with open('/dev/bone/pwm/1/a/enable', 'w') as filetowrite:
            filetowrite.write('1')

    def set_speed(self, value):
        # only allow the valur between 0 and 100
        value = max(0, min(100, value))
        self.current_output = value
        # convert output to pwm
        pwd = str(int(value * 4000 + 1500000))
        # set pwm
        with open('/dev/bone/pwm/1/a/duty_cycle', 'w') as filetowrite:
            filetowrite.write(pwd)
        return value

    def update(self):
        self.data["time"].append(time.time())
        # get speed
        self.current_speed = self.get_speed()
        # get error
        speed_error = self.target_speed - self.current_speed
        # get the pid output
        speed_pid = self.pid.get(speed_error)
        # add the pid output to current speed output
        self.current_output += speed_pid
        # set speed output
        self.current_output = self.set_speed(self.current_output)
        self.data["current_output"].append(self.current_output)


class Camera:
    # camera
    height = 240
    width = 320
    fps = 10
    # red paper
    red_threshold = 0.5 * height/2 * width
    # yolo
    model_size = 320
    model_path = "/home/debian/elec533/best-fp16-3.tflite"
    conf_threshold = 0.5
    iou_threshold = 0.45
    # save the intermidiate value
    data = dict()

    def __init__(self):
        # init camera
        self.cam = cv2.VideoCapture(2)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cam.set(cv2.CAP_PROP_FPS, self.fps)
        # init data
        self.data["red_mask"] = []
        self.data["k"] = []
        self.data["time"] = []
        self.data["frame"] = []
        self.data["stop"] = []
        # init model
        self.interpreter = tflite.Interpreter(
            model_path=self.model_path, num_threads=2)
        self.interpreter.allocate_tensors()
        self.output = self.interpreter.get_output_details()[0]
        self.input = self.interpreter.get_input_details()[0]

    def detect_stop(self, frame):
        # pre-process img
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.model_size, self.model_size))
        frame = frame.astype(np.float32)/255
        input_data = np.expand_dims(frame, axis=0)
        # process img
        self.interpreter.set_tensor(self.input['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output['index'])[0]
        # post-process img
        class_id = []
        boxes = []
        confs = []
        for i in range(output_data.shape[0]):
            confidence = output_data[i][4]
            # confidence threshold
            if confidence > self.conf_threshold:
                center_x = int(output_data[i][0] * self.width)
                center_y = int(output_data[i][1] * self.height)
                width = int(output_data[i][2] * self.width)
                height = int(output_data[i][3] * self.height)
                # convert box format for cv2.dnn.NMSBoxes
                left = center_x - width / 2
                top = center_y - height / 2
                class_id.append(0)
                confs.append(float(confidence))
                boxes.append([left, top, width, height])
        # NMS (not necessary)
        indices = cv2.dnn.NMSBoxes(
            boxes, confs, self.conf_threshold, self.iou_threshold)
        # self.display_box(boxes, confs, indices)
        # is there any stop sign
        for index in indices:
            i = index[0]
            box = boxes[i]
            return True
        return False

    def detect_red(self, hsv):
        # pink to red
        lower_red = np.array([140, 20, 40], dtype="uint8")
        upper_red = np.array([200, 255, 255], dtype="uint8")
        mask = cv2.inRange(hsv, lower_red, upper_red)
        # cv2.imshow("red",mask)
        # get number of red
        count = mask.sum()/255
        self.data["red_mask"].append(count)
        return (count > self.red_threshold)

    def detect_blue(self, hsv):
        # light blue to dark blue
        lower_blue = np.array([70, 90, 0], dtype="uint8")
        upper_blue = np.array([150, 255, 255], dtype="uint8")
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # cv2.imshow("blue",mask)
        return mask

    def display_box(self, boxes, confs, indices, line_color=(0, 255, 0), line_width=1):
        # background
        line_image = np.zeros([self.height, self.width, 3], np.uint8)

        for index in indices:
            i = index[0]
            left = int(boxes[i][0])
            top = int(boxes[i][1])
            width = int(boxes[i][2])
            height = int(boxes[i][3])
            # add box
            cv2.rectangle(line_image, (left, top), (left+width,
                          top+height), line_color, line_width)
            # add confidence
            cv2.putText(line_image, str(
                confs[i]), (left, top+height), cv2.FONT_HERSHEY_SIMPLEX, 1, line_color, line_width)
        cv2.imshow("stop", line_image)

    def display_lines(self, lines, line_color=(0, 255, 0), line_width=1):
        # background
        line_image = np.zeros([self.height, self.width, 3], np.uint8)

        for line in lines:
            for x1, y1, x2, y2 in line:
                # add line
                cv2.line(line_image, (x1, y1), (x2, y2),
                         line_color, line_width)
        cv2.imshow("lines", line_image)

    def detect_lines(self, mask):
        # detect edges
        edges = cv2.Canny(mask, 50, 100)
        # cv2.imshow("edges",edges)
        rho = 1
        theta = np.pi / 180
        min_threshold = 40
        # get lines
        line_segments = cv2.HoughLinesP(edges, rho, theta, min_threshold,
                                        np.array([]), minLineLength=20, maxLineGap=20)
        if line_segments is None:
            # no line!
            return 0
        # self.display_lines(line_segments)
        left = []
        right = []
        for line in line_segments:
            # calculate the theta
            theta = math.atan((line[0][3]-line[0][1]) /
                              (line[0][2]-line[0][0]+0.1))
            # we don't need horizontal lines
            if theta > 0.1:
                right.append(theta)
            elif theta < -0.1:
                left.append(theta)
        # average left
        left_theta = np.average(left) if left else 0
        # average right
        right_theta = np.average(right) if right else 0
        return (left_theta + right_theta)

    def update(self):
        # get image
        _, frame = self.cam.read()
        # save the image
        _, jpg = cv2.imencode(".jpg", frame)
        self.data["frame"].append(jpg)
        # cv2.imshow("frame",frame)
        self.data["time"].append(time.time())
        # get the hsv of half image for red detection and line detection
        hsv = cv2.cvtColor(frame[int(self.height/2):, :, :], cv2.COLOR_BGR2HSV)
        # red detection
        red = self.detect_red(hsv)
        # blue detection
        blue_mask = self.detect_blue(hsv)
        # line detection
        theta = self.detect_lines(blue_mask)
        # stop sign detection
        stop = self.detect_stop(frame)
        return [red, stop, theta]

    def close(self):
        # close camera
        self.cam.release()
        # output the image files
        for i, jpg in enumerate(self.data["frame"]):
            with open(str(i)+'.jpg', 'wb') as filetowrite:
                filetowrite.write(jpg.tobytes())


class Direction:

    target_direction = 0
    current_direction = 0
    current_output = 0
    # save the intermidiate value
    data = dict()

    def __init__(self, p, i, d):
        self.pid = PidClass2(p, i, d)
        self.data["current_direction"] = []
        self.data["current_output"] = []
        self.data["red"] = []
        self.data["time"] = []
        # init camera
        self.cam = Camera()

    def init_pwm(self):
        # period: 20000000
        with open('/dev/bone/pwm/1/b/period', 'w') as filetowrite:
            filetowrite.write('20000000')
        # duty_cycle: 7.5%
        with open('/dev/bone/pwm/1/b/duty_cycle', 'w') as filetowrite:
            filetowrite.write('1500000')
        # enable
        with open('/dev/bone/pwm/1/b/enable', 'w') as filetowrite:
            filetowrite.write('1')

    def set_direction(self, value):
        # value should be -100 t0 100
        value = max(-100, min(100, value))
        # convert value to pwm
        pwd = str(int(value * 3000 + 1500000))
        with open('/dev/bone/pwm/1/b/duty_cycle', 'w') as filetowrite:
            filetowrite.write(pwd)
        return value

    def update(self):
        # get results from camera
        red, stop, theta = self.cam.update()
        self.data["time"].append(time.time())
        self.data["current_direction"].append(theta)
        # get the pid output
        self.current_output = self.pid.get(theta)
        # set the pwm
        self.current_output = self.set_direction(self.current_output)
        self.data["current_output"].append(self.current_output)
        return red, stop

    def close(self):
        self.cam.close()


# %%
# init speed with target speed 200, p 0.05, i 0.01, d 0.01
speed = Speed(200, 0.05, 0.01, 0.01)
# init speed pwm
speed.init_pwm()
# init direction with p 60, i 0, d 0
direction = Direction(60, 0, 0)
# init direction pwm
direction.init_pwm()


# %%
# stop count how much loops the car should keep stop
stop_count = 0
# cooldown for red and stop signal
# don't stop again immediately after a stop
red_cooldown = 25
# how many red paper the car stops
red_count = 0
# how many stop sign the car stops
sign_count = 0
# total time, about 45 secs
run_loop = 300
for _ in range(run_loop):
    # is the car required to stop
    if stop_count:
        # count--
        stop_count -= 1
        # stop
        speed.set_speed(0)
    else:
        # update the speed
        speed.update()
    # update the direction, and get the red and stop sign
    red, stop = direction.update()
    # don't stop again immediately after a stop
    if red_cooldown > 0:
        red_cooldown -= 1
    # if this is the fisrt red paper
    elif red and red_count == 0:
        # stop 10 loops
        stop_count = 10
        # set cooldown to 30
        red_cooldown = 25
        # add the red count
        red_count = 1
    # if this is the second red paper
    elif red and red_count == 1:
        # stop
        speed.set_speed(0)
        # We made it! break the loop.
        break
    # if there is a stop sign
    if red_cooldown == 0 and stop and sign_count == 0:
        # stop 10 loops
        stop_count = 10
        # set cooldown to 30
        red_cooldown = 25
        # add the red count
        sign_count = 1

# stop forever
speed.set_speed(0)
# go straight
direction.set_direction(0)
