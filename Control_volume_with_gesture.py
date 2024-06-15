import cv2  #OpenCV 
import mediapipe as mp  #MediaPipe for hand tracking
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  #PyCaw for audio control
from comtypes import CLSCTX_ALL  #For comtypes to work with PyCaw
from ctypes import cast, POINTER  #For type casting

#to recognize the gesture based on the landmarks
def recognize_gesture(landmarks):
    #extracting landmarks for fingertips
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    #check hand gestures 
    if index_tip[1] < thumb_tip[1] and middle_tip[1] < thumb_tip[1]:
        if ring_tip[1] < thumb_tip[1] and pinky_tip[1] < thumb_tip[1]:
            return 'open_hand'  #open hand gesture
        else:
            return 'two_fingers'  #two fingers gesture
    elif index_tip[1] < thumb_tip[1]:
        return 'one_finger'  #one finger gesture
    else:
        return 'fist'  #fist gesture

#to set the system volume based on the gesture
def set_volume(gesture):
    #get the audio device and set its volume level
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    
    #set volume level according to recognized gesture
    if gesture == 'one_finger':
        volume.SetMasterVolumeLevelScalar(0.5, None)  # Set volume to 50%
    elif gesture == 'two_fingers':
        volume.SetMasterVolumeLevelScalar(0.3, None)  # Set volume to 30%
    elif gesture == 'open_hand':
        volume.SetMasterVolumeLevelScalar(1.0, None)  # Set volume to 100%
    elif gesture == 'fist':
        volume.SetMasterVolumeLevelScalar(0.0, None)  # Mute volume

#mediaPipe Hands and Drawing Utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  

#check if camera is opened
if not cap.isOpened():
    print("Error: Could not open video capture.")
else:
    #Main loop to capture and process video frames
    while True:
        #read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        #convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #process the frame to get hand landmarks
        result = hands.process(frame_rgb)

        #if hands are detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                #Draw landmarks and connections on the frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                #get landmarks as a list of tuples
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                
                #recognize hand gesture
                gesture = recognize_gesture(landmarks)
                
                #change system volume based on the recognized gesture
                set_volume(gesture)
                
                #Show the recognized gesture on the frame
                cv2.putText(frame, gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Frame', frame)

        #exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
