import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

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
worker_results_cache = {}

size = (1020,500)
#result = cv2.VideoWriter('filename.mp4', cv2.VideoWriter_fourcc(*'MJPG'),10, size)
stime=time.time()
while True:
    
    if(total_frames==count):
        break
    
    ret,frame = cap.read()   
    count += 1
    print(count)
    frame=cv2.resize(frame,(1020,500))
    #cv2.waitKey(40)
    
    if count % (30*2) == 0 or count==1:
        results=model1.predict(frame)
    prev_results = results  # Store the results for later use
    a=results[0].boxes.boxes
    px=pd.DataFrame(a).astype("float")
    for index,row in px.iterrows():
        class_id = int(row[5])
        conf = row[4]
        if conf > 0.3 and class_id == 0:
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
            worker_frame_key = f'{x1}{y1}{x2}_{y2}'
            if worker_frame_key in worker_results_cache:
                worker_results = worker_results_cache[worker_frame_key]
            else:
                worker_frame = frame[y1:y2, x1:x2, :]
                worker_results = model.predict(worker_frame)
                worker_results_cache[worker_frame_key] = worker_results
                print("Updating a new dictionary")
                print("worker_results_cache")
            worker_a = worker_results[0].boxes.boxes
            worker_px=pd.DataFrame(worker_a).astype("float")
            helmet_wearing = False
            vest_wearing = False
            mask_wearing = False
            for _, worker_row in worker_px.iterrows():
                label=""
                worker_class_id = int(worker_row[5])
                worker_conf = worker_row[4]
                if worker_conf > 0.25 and worker_class_id in [0,7]:
                    if worker_class_id == 0:
                        helmet_wearing = True
                    elif worker_class_id == 7:
                        vest_wearing = True
                #if worker_conf > 0.3 and worker_class_id in [1]:
                mask_wearing = True
            if helmet_wearing and vest_wearing and mask_wearing:
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

    #displaying the frame
        cv2.imshow("RGB", frame)

    #check for escape key
        if cv2.waitKey(1) & 0xFF == 27:
            break
etime=time.time()

print("Elapsed time: seconds", format(etime-stime))
#releasing the video capture object and closing all windows
cap.release()
cv2.destroyAllWindows()