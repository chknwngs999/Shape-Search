# source: https://learnopencv.com/edge-detection-using-opencv/

import cv2


def detect_shape(contour):
    # Implement the logic to classify the shape based on features extracted from the contour
    # Return the shape label (e.g., 'triangle', 'rectangle', 'pentagon', etc.)
    # Approximate the contour to reduce the number of vertices
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Get the number of vertices (sides) of the shape
    num_vertices = len(approx)

    if num_vertices == 3:
        return "Triangle"
    elif num_vertices == 4:
        # Check if the shape is a rectangle or square
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            return "Square"
        else:
            return "Rectangle"
    elif num_vertices == 5:
        return "Pentagon"
    else:
        return "Unknown"


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.GaussianBlur(frame, ksize=(5, 5), sigmaX=0, sigmaY=0)
    # Image preprocessing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, 50, 150)

    # Contour detection
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Skip small contours to avoid noise

        if cv2.contourArea(contour) > 800:
            shape_label = detect_shape(contour)

            # Draw the shape label on the frame
            moments = cv2.moments(contour)
            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])
            cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
            cv2.putText(frame, shape_label, (centroid_x, centroid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with shapes
    cv2.imshow('Edges?', edges)
    cv2.imshow('Shape Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(33) == 27:
        break

cap.release()
cv2.destroyAllWindows()
