
## This code was adapted from Real-time Object Detection with YOLO and Webcam: Enhancing Your Computer Vision Skills by Dipankar Medhi (available at https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993)
from ultralytics import YOLO
import cv2
import math
import mediapipe as mp
import sys
from gtts import gTTS
import os
from playsound import playsound
##mytext = 'Keep calm. Emergency services have been contacted'
##myobj = gTTS(text=mytext, lang='en', slow=False)
##myobj.save("contacted.mp3")
##myobj = gTTS(text = "Great! Head to the nearest safe shelter as soon as possible", lang='en', slow=False)
##myobj.save("shelter.mp3")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
mpDraw = mp.solutions.drawing_utils

mpPose = mp.solutions.pose
pose = mpPose.Pose()
pTime = 0

model = YOLO("yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
tiltaud = False
tiltdet = False
done = False
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    text = "Tilt: NONE"
    mytilt = 0
    if results.pose_landmarks:
        if not tiltaud:
            playsound("./tilthead.mp3")
            tiltaud = True
        mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        enum = enumerate(results.pose_landmarks.landmark)
        imp = []
        for id, lm in enum:
            h,w,c = img.shape
            cx = int(lm.x*w)
            cy = int(lm.y*h)
            cv2.circle(img, (cx,cy),5, (255,0,0), cv2.FILLED)
            if id == 0 or id == 9 or id ==10:
                imp.append([cx,cy])
        mid = [(imp[1][0]+imp[2][0])//2, ((-imp[1][1])+(-imp[2][1]))//2]
        grad = (mid[1] - (-imp[0][1])) /  ((mid[0] - imp[0][0])+0.000001)
        if grad < 2 and grad > 0:
            text = "Tilt: LEFT"
            mytilt = -1
        elif grad > -2 and grad < 0:
            text = "Tilt: RIGHT"
            mytilt = 1
         
    results2 = model(img, stream=True)
    
    for r in results2:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (166, 218, 149), 3)

            
            confidence = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls].upper() + f" Confidence: {confidence} {text}", org, font, fontScale, color, thickness)

    if tiltaud and not tiltdet and text != "Tilt: NONE":
        tiltdet = True
        if text == "Tilt: RIGHT":
            playsound("./contacted.mp3")
        else:
            playsound("./audio.mp3")
            
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
