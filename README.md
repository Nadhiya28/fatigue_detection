# Fatigue Detection Web App
This is a web application that detects driver fatigue using eye aspect ratio (EAR) and mouth aspect ratio (MAR) analysis from video input or webcam.

## Features
- Detects drowsiness and yawning to alert fatigue.
- Supports video upload or live webcam feed.
- Real-time display of EAR and MAR.
- Plays alert sound when fatigue is detected.
- Tracks and logs fatigue events with timestamps.
- Allows downloading fatigue event report as CSV.

## Requirements
- Python 3.7+
- Packages: streamlit, opencv-python, dlib, numpy, pandas, scipy

## Setup Instructions
1. Clone or download this repository.
2. Download the shape predictor file from [dlib website](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
3. Extract the file and place `shape_predictor_68_face_landmarks.dat` in the project folder.
4. Install dependencies:
pip install -r requirements.txt
5. Run the app:
streamlit run app.py


## Notes
- The `shape_predictor_68_face_landmarks.dat` file (~100MB) is not included due to size. Please download it separately.
- Tested on Windows 10 and Ubuntu.

## License
This project is open-source and free to use for educational purposes.

