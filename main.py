# customizing - comment/uncomment medianBlur/bilateralFilter
#

import cv2
import numpy as np

# https://stackoverflow.com/questions/60674501/how-to-make-black-background-in-cv2-puttext-with-python-opencv


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


vid = cv2.VideoCapture(0)


while True:
    ret, frame = vid.read()
    # frame = cv2.medianBlur(frame, 7)
    frame = cv2.bilateralFilter(frame, 9, 75, 75)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(
        gray, 127, 255, cv2.THRESH_BINARY, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    """threshold = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)"""

    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area, largest to smallest, and filter out small areas
    largest_contours = contours = sorted(contours, key=lambda x: cv2.contourArea(
        x), reverse=True)
    for i in range(len(largest_contours)):
        if cv2.contourArea(largest_contours[i]) <= 20:
            largest_contours = largest_contours[1:i]
            break

    # for contour in contours:
    for contour in largest_contours:
        # cv2.approxPloyDP() function to approximate the shape
        approx_sides = cv2.approxPolyDP(
            contour, 0.02 * cv2.arcLength(contour, True), True)

        # filter for quadrilaterals
        filter_shapes = [3, 4, 5]
        if len(approx_sides) in filter_shapes:
            # using drawContours() function
            cv2.drawContours(frame, [contour], 0, (0, 0, 255), 3)
            # cv2.drawContours(threshold, [contour], 0, (0, 0, 255), 3)

            # finding center point of shape
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])

                # putting shape name at center of each shape
                draw_text(frame, str(len(approx_sides)), (x, y))

    cv2.imshow('frame', frame)
    """cv2.imshow('gray', gray)
    cv2.imshow('threshold', threshold)"""

    if cv2.waitKey(33) == 27:
        break

vid.release()
cv2.destroyAllWindows()
