Hey there! Welcome to Gesture-Controlled Volume, a fun project that lets you control your system volume using hand gestures captured from your webcam.

How it Works:
Imagine you're sitting in front of your computer, and instead of using your mouse or keyboard to adjust the volume, you can simply make hand gestures!

How to Use:
Install Dependencies:
Before you get started, make sure you have the following dependencies installed:

OpenCV (cv2)
MediaPipe (mediapipe)
PyCaw (pycaw)
Comtypes (comtypes)
Run the Script:

Simply run the provided Python script gesture_volume_control.py.
This script will access your webcam to detect your hand gestures.
Control Volume:

Once the script is running, point your hand towards the camera.
Open your hand to increase the volume to the maximum.
Make a fist to mute the volume.
Show one finger to set the volume to 50%.
Show two fingers to set the volume to 30%.
Exit:

Press the 'q' key to exit the program.
What's Happening:
Hand Detection:
The program uses the MediaPipe library to detect your hand landmarks from the video stream captured by your webcam.

Gesture Recognition:
It recognizes different hand gestures based on the position of your fingertips. For example:

Open hand gesture increases the volume.
Fist gesture mutes the volume.
Showing one finger sets the volume to 50%.
Showing two fingers sets the volume to 30%.
Volume Control:
PyCaw library is used to control the system volume. Depending on the recognized gesture, the volume is adjusted accordingly.

Tips:
Lighting: Make sure you have good lighting in your room for better hand detection.
Background: A plain background helps in accurate hand detection.
Hand Position: Keep your hand in front of the webcam for better recognition.
