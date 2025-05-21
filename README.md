💤 Driver Drowsiness Detection System
🚗 Overview
This project is a real-time Driver Drowsiness Detection System that uses OpenCV, Dlib, and Python to monitor a driver’s alertness. By analyzing facial landmarks—particularly the eye aspect ratio (EAR)—the system can detect when a driver is falling asleep or becoming drowsy. If signs of drowsiness are detected, an audio alert is triggered to prevent potential accidents and enhance road safety.

🧠 Features
Real-time eye monitoring using a webcam.

Calculates Eye Aspect Ratio (EAR) to detect eye closure.

Triggers an audible beep alert if the driver's eyes remain closed beyond a safe threshold.

Uses facial landmark detection with Dlib's pretrained model.

🛠️ Tech Stack
Python

OpenCV

Dlib

Imutils

Scipy

Winsound/OS (for cross-platform sound alerts)

📁 Project Structure
DriverDrowsinessDetection/
│
├── driver_drowsiness.py                  # Main application script
├── shape_predictor_68_face_landmarks.dat # Dlib pre-trained model (downloaded separately)
├── README.md                             # Project documentation
