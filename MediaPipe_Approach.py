import cv2
import pandas as pd
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import time
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
model1 = YOLO('yolov8m.pt')
model=YOLO('best.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)



cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('22 5sec.mp4')
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
my_file = open("classes.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)
count=0
prev_results = None
worker_results_dict = {}

size = (1020,500)
#result = cv2.VideoWriter('filename.mp4', cv2.VideoWriter_fourcc(*'MJPG'),10, size)
stime=time.time()
while True:    
    ret, frame = cap.read()   
    count += 1
    print(count)
    if(total_frames==count):
        break
    frame = cv2.resize(frame, (1020,500))
    #cv2.waitKey(40)
    
    if count % (24*0.5) == 0 or count==1:
        results = model.predict(frame)
        prev_results = results  # Store the results for later use
    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")
    
    for index,row in px.iterrows():
        class_id = int(row[5])
        conf = row[4]
        if conf > 0.5 and class_id == 5:
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1) 
            worker_frame = frame[y1:y2, x1:x2, :]
            imgRGB = cv2.cvtColor(worker_frame, cv2.COLOR_BGR2RGB)
            resu = pose.process(imgRGB)
            if count % (24*0.5) == 0 or count==1:
                worker_results = model.predict(worker_frame)
                worker_results_dict[(x1, y1)] = worker_results  # Store the results for later use
            else:
                worker_results = worker_results_dict.get((x1, y1), None)
            worker_a = worker_results[0].boxes.boxes
            worker_px=pd.DataFrame(worker_a).astype("float")
            helmet_wearing = False
            vest_wearing = False
            mask_wearing = False
            if resu.pose_landmarks is not None:
                for id, lm in enumerate(resu.pose_landmarks.landmark):
                    if id in[4,12,9,1,11,24,23,10,0]:
                        h, w, c = worker_frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(worker_frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                        print(id)
                        for index, row in worker_px.iterrows():
                            class_id = int(row[5])
                            if class_id in [0, 7]:  # helmet or vest bounding box
                                if cx > row[0] and cx < row[2] and cy > row[1] and cy < row[3]:
                                    # Body part detected inside bounding box
                                    if class_id == 0:
                                        helmet_wearing = True
                                        print("hello")
                                    elif class_id == 7:
                                        vest_wearing = True
                                    elif class_id == 1:  # mask bounding box
                                        if cx > row[0] and cx < row[2] and cy > row[1] and cy < row[3]:
                                            # Body part detected inside mask bounding box
                                            mask_wearing = True
            if helmet_wearing and vest_wearing or mask_wearing:
                label = "yes"
                color = (0, 255, 0)
            else:
                label="no "
                if not helmet_wearing:
                    label+="helmet "
                if not vest_wearing:
                    label+="vest "
                if not mask_wearing:
                    label+="mask"
                color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)


                # Iterate through landmarks and check for intersection with safety gear bounding boxes
# Loop through all landmarks detected by MediaPipe Pose
        cv2.imshow("RGB", frame)

    #check for escape key
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()

etime=time.time()

print("Elapsed time: {:.2f} seconds", format(etime-stime))

# Display the worker frame with safety gear bounding boxes
cv2.imshow("Worker Frame", frame)
cv2.waitKey(0)