import cv2
import numpy as np

def detect_hand(frame):
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the HSV range for skin color
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin colors
    mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

    # Apply morphological transformations to filter out noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    return mask

def track_hand():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Detect hand in the frame
        mask = detect_hand(frame)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour
            max_contour = max(contours, key=cv2.contourArea)

            # Draw the contour
            cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)

            # Get the convex hull of the contour
            hull = cv2.convexHull(max_contour, returnPoints=False)
            if hull is not None:
                # Find convexity defects
                defects = cv2.convexityDefects(max_contour, hull)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(max_contour[s][0])
                        end = tuple(max_contour[e][0])
                        far = tuple(max_contour[f][0])
                        depth = d / 256.0  # Convert to floating-point depth

                        # Filter out small defects by depth
                        if depth > 10:  # You can adjust this threshold
                            # Draw the defects points
                            cv2.circle(frame, start, 5, [0, 0, 255], -1)
                            cv2.circle(frame, end, 5, [0, 0, 255], -1)
                            cv2.circle(frame, far, 5, [255, 0, 0], -1)
                            # Draw the lines connecting start, end, and far points
                            cv2.line(frame, start, end, [0, 255, 0], 2)
                            cv2.line(frame, start, far, [0, 255, 0], 2)
                            cv2.line(frame, end, far, [0, 255, 0], 2)

        # Display the resulting frame
        cv2.imshow('Hand Tracking', frame)

        # Press 'q' to exit the webcam feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_hand()
