# unnecessary?

import cv2

camera = cv2.VideoCapture(0)

stream = True
while stream:
    # Read the camera frame
    # device.speed(velocity=60, acceleration=60)
    # returns (1) bool if we get frames or not (2) the frame
    success, frame = camera.read()
    if not success:
        break
    else:
        # HSV ex at https://stackoverflow.com/questions/47483951/how-to-define-a-threshold-value-to-detect-only-green-colour-objects-in-an-image/47483966#47483966
        # returns (480, 640, 3) np arrays
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # I think this might be frame's masked counterpart
        color_mask = cv2.inRange(hsv_frame, low_color, high_color)
        # Countours is a list of rank=3 nparrays of varying dimensions
        contours, _ = cv2.findContours(
            color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(
            x), reverse=True)  # Sort contours by area, largest to smallest
        # Guiding Quesion 1
        # Establish priority for "large" contours which are closest to center
