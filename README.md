Eye Tracking using OpenCV
This repository contains Python code to detect and track eyes using OpenCV. The code utilizes pre-trained Haar Cascade classifiers for face and eye detection.

Requirements
Python 3.x
OpenCV (cv2) library
Installation
Clone this repository to your local machine using the following command:
bash
Copy code
git clone https://github.com/Abdulkadhem/eye-tracking.git
Navigate to the cloned directory:
bash
Copy code
cd eye-tracking
Install the required dependencies using pip:
Copy code
pip install -r requirements.txt
Usage
Run the eye_tracking.py script to start the eye tracking application:
Copy code
python eye_tracking.py
Position yourself in front of the webcam.
The application will detect your face and track your eyes in real-time.
Press 'q' to exit the application.
Troubleshooting
If you encounter issues with the webcam or face/eye detection, make sure your camera is properly connected and accessible by OpenCV.
Adjust the scaleFactor, minNeighbors, and minSize parameters in the code for better face and eye detection based on your environment and requirements.
Contributing
Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

