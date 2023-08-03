import time
import numpy as np
import cv2
from flask import Flask, render_template, Response, redirect, url_for
from serial.tools import list_ports
import time
import pydobot


def draw_text(img, text, pos,
              font=cv2.FONT_HERSHEY_SIMPLEX,
              font_scale=1,
              font_thickness=2,
              text_color=(255, 255, 255),
              text_color_bg=(0, 0, 0)
              ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1),
                font, font_scale, text_color, font_thickness)

    return text_size
# Get cameras function from utils.py in RETA


def get_cameras():
    available_cams = []
    for camera_idx in range(10):
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            available_cams.append(camera_idx)
            cap.release()
        else:
            # suppress warnings from cv2
            print('\033[A' + ' '*158 + '\033[A')
    return available_cams


# dobot
available_ports = list_ports.comports()
print(f'available ports: {[x.device for x in available_ports]}')
available_cams = get_cameras()
print("available cameras: ", available_cams)
port = available_ports[0].device  # DOBOT PORT NAME: /dev/ttyACM0
device = pydobot.Dobot(port=port, verbose=False)
device.speed(velocity=60, acceleration=60)
time.sleep(5)

# App
app = Flask(__name__)
camera_side = cv2.VideoCapture(2)


class pos_keeper():
    def __init__(self, j1, j2, j3, j4):
        self.j1 = j1
        self.j2 = j2
        self.j3 = j3
        self.j4 = j4


Pos = pos_keeper(0, 0, 0, 0)


@app.context_processor
def context_processor():
    return dict(j1=Pos.j1,
                j2=Pos.j2,
                j3=Pos.j3,
                j4=Pos.j4)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def generate_frames(mask: "bool" = False):
    '''
    Feed function int to select camera for cv2 to open

    !!! Consider only allowing movements for odd numbered frames
        - Simple but hacky way to get movements to slow down which should make 
          it easier for the dobot to register objects in view
    '''
    camera = cv2.VideoCapture(0)
    # Make resolution simpler to boost performance
    camera.set(3, 480)  # switch width from 640 to 480
    camera.set(4, 320)  # switch height from 480 to 320

    _, frame = camera.read()
    rows, cols, _ = frame.shape  # rows, cols, channels
    x_center = int((cols) / 2)  # x_center of screen
    x_medium = int((cols) / 2)
    y_center = int((rows) / 2)  # y_enter of screen
    old_y_center = int((rows) / 2)
    y_medium = int((rows) / 2)
    # Reset Dobot Magician Lite position

    j1, j2, j3, j4 = 0, 0, 0, 0
    first_phase = True  # First phase consists of centering screen to object
    # Once first phase is over (screen centered on object) dobot will then grab object

    device.rotate_joint(j1, j2, j3, j4)
    # Loop for colored-object tracking
    count = 0  # Counter for how many times arm converged object to center
    stream = True
    found = False
    scan = 0  # Once no objects appear as targets, increment by 1 to perform a total of 4 scans
    scan_frames = 3  # Frame buffer used for scanning

    # Refer to this for post for how to input colors in HSV format: https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
    low_color = np.array([161, 155, 84])  # Red in HSV format
    high_color = np.array([179, 255, 255])

    frame_num = 0

    while stream:
        # Read the camera frame
        # device.speed(velocity=60, acceleration=60)
        # returns (1) bool if we get frames or not (2) the frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # crops frames so that top of mat can easily be adjusted to fit snugly in frame
            frame = frame[5:, 63:287]
            frame = cv2.bilateralFilter(frame, 9, 75, 75)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            _, threshold = cv2.threshold(
                gray_frame, 127, 255, cv2.THRESH_BINARY, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
            contours, _ = cv2.findContours(
                threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(
                x), reverse=True)[1:]  # Sort contours by area, largest to smallest, remove full screen contour
            for c in range(len(contours)):
                if cv2.contourArea(contours[c]) <= 20:
                    contours = contours[1:c]
                    break
            for c in range(len(contours)):
                if cv2.contourArea(contours[c]) <= 200:
                    contours = contours[c:]
                    break
            # Guiding Quesion 1
            # Establish priority for "large" contours which are closest to center
            try:
                maxArea = cv2.contourArea(contours[0])  # Largest area

            except:  # If no area can be found from the "try"
                maxArea = 0
                found = True
            error = 0.2
            largeContourPairs = []
            for contour in contours:  # Compile large contours
                approx_sides = cv2.approxPolyDP(
                    contour, 0.02 * cv2.arcLength(contour, True), True)
                filter_shapes = [3, 4, 5]
                if len(approx_sides) in filter_shapes:
                    area = cv2.contourArea(contour)
                    cv2.drawContours(frame, [contour], 0, (0, 0, 255), 3)
                    if area <= maxArea + maxArea and area >= maxArea - maxArea:  # If contour is within area boundaries
                        (x, y, w, h) = cv2.boundingRect(contour)
                        # middle line must be int since pixels are ints
                        x_medium = int((x + x + w) / 2)
                        y_medium = int((y + y + h) / 2)
                        # 2d distance calc for object centroid to center of screen
                        # 2d distance calc for object centroid to center of screen
                        dist = np.sqrt(np.power(x_medium - 112, 2) +
                                       np.power(y_medium - old_y_center, 2))
                        largeContourPairs.append((contour, dist))
                        draw_text(frame, str(len(approx_sides)),
                                  (x_medium, y_medium))
                        found = False
            largeContourPairs = sorted(
                largeContourPairs, key=lambda largeContourPairs: largeContourPairs[1])  # Sort by dist
            # Now create crosshair to home in on object
            # vertical pixel distance from top of screen to home x position on grid
            x_160mm_pxdist = 110
            for cnt, dist in largeContourPairs:  # iterate over contour frames
                (x, y, w, h) = cv2.boundingRect(cnt)
                # middle line must be int since pixels are ints
                x_medium = int((x + x + w) / 2)
                y_medium = int((y + y + h) / 2)
                # 2d distance calc from object centroid to bottom center of screen
                # 2d distance calc from object centroid to approximately the pole of j1 rotation
                # 2d distance calc from object centroid to approximately the pole of j1 rotation
                r_medium = np.sqrt(np.power(x_medium - 112, 2) + np.power(
                    y_medium - (x_160mm_pxdist + (250 * x_160mm_pxdist / 160)), 2))
                # polar angle of object centroid with pole at approximately the pole of j1 rotation and polar axis being vertical line, left half is positive, right half negative
                theta_medium = -np.degrees(np.arctan((x_medium - 112) / (
                    (x_160mm_pxdist + (250 * x_160mm_pxdist / 160)) - y_medium)))
                # position of block in image coordinates
                print(x_medium, y_medium)
                if frame_num < 10:
                    x1, y1, z1, r1, j11, j21, j31, j41 = device.pose()
                    # rotates j1 to ready arm for grab (makes it look like the arm is honing in)
                    # rotates j1 to ready arm for grab
                    device.rotate_joint(1.3 * theta_medium, 0, 0, 0)
                    time.sleep(1/60)
                    frame_num += 1
                elif frame_num == 10:  # grabs after 30 frames
                    # converts the image coordinates to Dobot coordinates
                    to_x = 410 - ((160/x_160mm_pxdist) * y_medium)
                    to_y = (180/112) * (112 - x_medium)
                    # print(to_x, to_y)
                    device.move_to(x=to_x, y=to_y, z=0, r=r1, wait=True)
                    grab()
                    j1, j2, j3, j4 = 0, 0, 0, 0
                    frame_num = 0
                # print(x_medium, y_medium)
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Just interested in first sorted (used to be on biggest, now it's on closest) rectangle/lines
                break
            # Code for crosshair
            cv2.line(frame, (x_medium, 0), (x_medium, 480), (0, 255, 0), 1)
            cv2.line(frame, (0, y_medium), (640, y_medium), (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX

            try:  # Some error occurs when adding dist line that I'm too lazy to fix lol
                # Error seems only to occur AFTER First phase is completed and grabbing centering begins (dist line disappears; highly noticeable from video)
                cv2.line(frame, (x_medium, y_medium), (112, int(
                    x_160mm_pxdist + (250 * x_160mm_pxdist / 160))), (255, 0, 255), 1)  # Dist line
                cv2.putText(img=frame, text="(r, theta): ",  org=(x_medium + 20,  y_medium), fontFace=font, fontScale=0.3,
                            color=(0, 255, 255))
                cv2.putText(img=frame, text=f"({round(r_medium)}, {theta_medium:.3})",  org=(x_medium + 20,  y_medium + 10), fontFace=font, fontScale=0.3,
                            color=(0, 255, 255))
                cv2.putText(img=frame, text="(x, y): ",  org=(x_medium + 20,  y_medium + 20), fontFace=font, fontScale=0.3,
                            color=(0, 255, 255))
                cv2.putText(img=frame, text=f"({x_medium}, {y_medium})",  org=(x_medium + 20,  y_medium + 40), fontFace=font, fontScale=0.3,
                            color=(0, 255, 255))
            except:
                cv2.putText(img=frame, text=f"DIST: X.XX px",  org=(112 + 20,  old_y_center), fontFace=font, fontScale=0.3,
                            color=(0, 255, 255))
                pass
            # Encodes frame into memory buffer
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            output_frame = b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            # m_ret, m_buffer = cv2.imencode(".jpg", color_mask) # Encodes masked frame into memory buffer
            # color_mask = m_buffer.tobytes()
            # output_color_mask = b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + color_mask + b'\r\n'
            yield (output_frame)


def grab():
    '''
    Frames are basically frozen during this,
    consider using multiprocess or something to have the 
    grabbing function run while frame generator passes 
    in order to yield frames while grabbing is occurring
    '''
    global color
    x, y, z, r, j1, j2, j3, j4 = device.pose()
    device.grip(False)
    device.move_to(x=x, y=y, z=-24, r=r)
    time.sleep(1)
    device.grip(True)
    time.sleep(1)
    device.move_to(x=x, y=y, z=15, r=r)
    device.rotate_joint(j1=80, j2=30, j3=0, j4=0)
    time.sleep(1)
    device.grip(False)
    time.sleep(1)
    device.suck(False)


def generate_frames_side():
    '''
    Feed function int to select camera for cv2 to open
    '''

    # Loop for colored-object tracking
    while True:
        # Read the camera frame
        # returns (1) bool if we get frames or not (2) the frame
        success, frame = camera_side.read()
        if not success:
            break
        else:
            # Encodes frame into memory buffer
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            output_frame = b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            yield (output_frame)


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_side')
def video_feed_side():
    return Response(generate_frames_side(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_feed2')
# def video_feed2():
#     return Response(generate_frames(2, True),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    port = 5000
    app.run(host='0.0.0.0', port=port, debug=False)
    device.rotate_joint(0, 0, 0, 0)
