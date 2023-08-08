# https://learnopencv.com/blob-detection-using-opencv-python-c/

import cv2
import numpy as np

# https://stackoverflow.com/questions/60674501/how-to-make-black-background-in-cv2-puttext-with-python-opencv


def draw_text(img, text, pos,
              font=cv2.FONT_HERSHEY_SIMPLEX,
              font_scale=1,
              font_thickness=1,
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


vid = cv2.VideoCapture(0)


while True:
    ret, frame = vid.read()
    frame = cv2.bilateralFilter(frame, 9, 75, 75)

    # Convert the frame to grayscale and apply adaptive thresholding
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    threshold = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Using blob detection to find shapes
    detector_params = cv2.SimpleBlobDetector_Params()
    detector_params.filterByCircularity = True
    detector_params.minCircularity = 0.87
    detector_params.filterByArea = True
    detector_params.minArea = 20
    detector_params.maxArea = 5000
    detector = cv2.SimpleBlobDetector_create(detector_params)
    keypoints = detector.detect(threshold)

    print(keypoints)
    for keypoint in keypoints:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        size = int(keypoint.size)
        cv2.drawMarker(frame, (x, y), (0, 0, 255),
                       cv2.MARKER_STAR, markerSize=size)

    cv2.imshow('frame', frame)
    cv2.imshow('threshold', threshold)

    if cv2.waitKey(33) == 27:
        break

vid.release()
cv2.destroyAllWindows()
