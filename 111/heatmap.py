import cv2
import numpy as np

def webcam_to_heatmap():
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

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Normalize the grayscale image to the range [0, 255]
        normalized_gray = cv2.normalize(gray_frame, None, 0, 255, cv2.NORM_MINMAX)

        # Apply a color map to the normalized grayscale image
        heatmap = cv2.applyColorMap(normalized_gray, cv2.COLORMAP_JET)

        # Display the resulting heatmap
        cv2.imshow('Webcam Heatmap', heatmap)

        # Press 'q' to exit the webcam feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_to_heatmap()
