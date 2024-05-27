````markdown
# Eye Tracking using OpenCV

This repository contains Python code to detect and track eyes using OpenCV. The code utilizes pre-trained Haar Cascade classifiers for face and eye detection.

## Requirements

```bash
# Python 3.x
# OpenCV (cv2) library
```

## Installation

```bash
# Clone this repository to your local machine:
git clone https://github.com/Abdulkadhem/eye-tracking.git

# Navigate to the cloned directory:
cd eye-tracking

# Install the required dependencies using pip:
pip install -r requirements.txt
```

## Usage

```bash
# Run the eye_tracking.py script to start the eye tracking application:
python eye_tracking.py
```

1. Position yourself in front of the webcam.
2. The application will detect your face and track your eyes in real-time.
3. Press 'q' to exit the application.

## Troubleshooting

```bash
# If you encounter issues with the webcam or face/eye detection, make sure your camera is properly connected and accessible by OpenCV.
# Adjust the `scaleFactor`, `minNeighbors`, and `minSize` parameters in the code for better face and eye detection based on your environment and requirements.
```

## Contributing

```bash
# Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.
```

## License

```bash
# This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

