import cv2

vid = cv2.VideoCapture(0)

while True:
    idk, frame = vid.read()
    frameGry = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, thrash = cv2.threshold(
        frameGry, 240, 255, cv2.CHAIN_APPROX_NONE)  # difference
    contours, hierarchy = cv2.findContours(
        thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # None vs Simple

    # cv2.drawContours(frameGry, contours, -1, (0, 255, 0), 3)
    print(len(contours))
    for contour in contours:
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)  # change the multiplier?

        cv2.drawContours(frame, [approx], 0, (204, 255, 0), 5)

        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5

        if len(approx) == 3:
            cv2.putText(frame, "Triangle", (x, y),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (204, 255, 0))
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspectRatio = float(w)/h
            # print(aspectRatio)
            if aspectRatio >= 0.95 and aspectRatio < 1.05:
                cv2.putText(frame, "square", (x, y),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (204, 255, 0))

            else:
                cv2.putText(frame, "rectangle", (x, y),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (204, 255, 0))

        elif len(approx) == 5:
            cv2.putText(frame, "pentagon", (x, y),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (204, 255, 0))
        elif len(approx) == 10:
            cv2.putText(frame, "decagon", (x, y),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (204, 255, 0))
        """else:
            cv2.putText(frame, "circle", (x, y),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (204, 255, 0))"""

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

vid.release()
cv2.destroyAllWindows()
