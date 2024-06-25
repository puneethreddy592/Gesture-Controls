<div align="center">
  <h1>Gesture Volume Control Using OpenCV and MediaPipe</h1>
  <img alt="output" src="images/output.gif" />
</div>
This project uses OpenCV and MediaPipe to control system volume, take screenshots, and play/pause media using hand gestures.

💾 REQUIREMENTS
opencv-python
mediapipe
comtypes
numpy
pycaw
pyautogui
bash
Copy code
pip install -r requirements.txt
MEDIAPIPE
<div align="center">
  <img alt="mediapipeLogo" src="images/mediapipe.png" />
</div>
MediaPipe offers open source cross-platform, customizable ML solutions for live and streaming media.

Hand Landmark Model
After the palm detection over the whole image, the subsequent hand landmark model performs precise keypoint localization of 21 3D hand-knuckle coordinates inside the detected hand regions via regression. The model learns a consistent internal hand pose representation and is robust even to partially visible hands and self-occlusions.

Source: MediaPipe Hands Solutions

<div align="center">
    <img alt="mediapipeLogo" src="images/hand_landmarks_docs.png" height="200" />
    <img alt="mediapipeLogo" src="images/htm.jpg" height="360" width="640" />
</div>
📝 CODE EXPLANATION
Importing Libraries
python
Copy code
import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pyautogui
Solution APIs
python
Copy code
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
Volume Control Library Usage
python
Copy code
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
Getting Volume Range
python
Copy code
volRange = volume.GetVolumeRange()
minVol, maxVol, volBar, volPer = volRange[0], volRange[1], 400, 0
Setting up Webcam using OpenCV
python
Copy code
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
Using MediaPipe Hand Landmark Model for Identifying Hands
python
Copy code
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:

  while cap.isOpened():
    success, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            img,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
Finding Hand Landmarks Position
python
Copy code
lmList = []
if results.multi_hand_landmarks:
    myHand = results.multi_hand_landmarks[0]
    for id, lm in enumerate(myHand.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmList.append([id, cx, cy])
Assigning Variables for Thumb and Index Finger Position
python
Copy code
if len(lmList) != 0:
    x1, y1 = lmList[4][1], lmList[4][2]
    x2, y2 = lmList[8][1], lmList[8][2]
Marking Thumb and Index Finger using cv2.circle() and Drawing a Line between Them using cv2.line()
python
Copy code
cv2.circle(img, (x1, y1), 15, (255, 255, 255), cv2.FILLED)
cv2.circle(img, (x2, y2), 15, (255, 255, 255), cv2.FILLED)
cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
length = math.hypot(x2 - x1, y2 - y1)
if length < 50:
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
Converting Length Range into Volume Range using numpy.interp()
python
Copy code
vol = np.interp(length, [50, 220], [minVol, maxVol])
Changing System Volume using volume.SetMasterVolumeLevel() Method
python
Copy code
volume.SetMasterVolumeLevel(vol, None)
volBar = np.interp(length, [50, 220], [400, 150])
volPer = np.interp(length, [50, 220], [0, 100])
Drawing Volume Bar using cv2.rectangle() Method
python
Copy code
cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 0), 3)
cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)
Displaying Output using cv2.imshow Method
python
Copy code
cv2.imshow('Img', img)
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
Closing Webcam
python
Copy code
cap.release()
cv2.destroyAllWindows()
👋 Gesture Controls
Switching Modes
Gesture: Pinky and thumb up, all other fingers down
Description: Switches between Volume, Screenshot, and Media Control modes
Volume Control Mode
Gesture: Normal behavior with index and thumb distance
Description: Adjusts the system volume
Lock/Unlock Volume:
Gesture: Touch middle finger and thumb, all other fingers down
Description: Locks and unlocks the volume
Screenshot Mode
Gesture: Touch index and thumb, all other fingers down
Description: Takes a screenshot
Media Control Mode
Gesture: Touch index and thumb, all other fingers down
Description: Plays/pauses media
📬 Contact
If you want to contact me, you can reach me through the below handles.

<a href="https://twitter.com/prrthamm"><img src="https://upload.wikimedia.org/wikipedia/fr/thumb/c/c8/Twitter_Bird.svg/1200px-Twitter_Bird.svg.png" width="25">@prrthamm</img></a>  
<a href="https://www.linkedin.com/in/pratham-bhatnagar/"><img src="https://www.felberpr.com/wp-content/uploads/linkedin-logo.png" width="25"> Pratham Bhatnagar</img></a>
