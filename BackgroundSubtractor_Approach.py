import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

def is_closer(num, target, compare):
    """
    Check if a given number is closer to the target number than the compare number
    """
    return abs(num - target) < abs(num - compare)


# Load the YOLO models
#model1 = YOLO('yolov8m.pt')
model = YOLO('best.pt')

fgbg = cv2.createBackgroundSubtractorMOG2()


# Define a callback function to get the RGB values of the pixel where the mouse is moved over
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


# Create a window and set the mouse callback function
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open the video file
cap = cv2.VideoCapture('22 5sec.mp4')
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# Read the list of classes from a file
my_file = open("classes.txt", "r")
data = my_file.read()
class_list = data.split("\n")
print("Class List:")
print(class_list)

count = 0
prev_results = None
worker_results_dict = {}
flag = False
co = {}

size = (1020,500)
stime=time.time()
while True:
    
    # Read a frame from the video
    ret, frame = cap.read()
    ret, image = cap.read()
    count += 1
    print(count)
    if(total_frames-5==count):
        break
    
    print("count: "+ str(count))
    image = cv2.resize(frame, (1020, 500))
    # Increment the frame counter
    
    
    
    
    # Resize the frame to the desired size
    frame = cv2.resize(frame, size)

    # Wait for a key press
    #cv2.waitKey(40)

    # Run the YOLO model on the frame
    if flag or count % (30*2) == 0 or count ==1:
        results = model.predict(frame)
        # prev_results = results
        flag = False
        # Store the results for later use

    fgmask = fgbg.apply(image)
    cv2.imshow('fgmask', fgmask)

    # Threshold the motion mask to remove noise
    thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("total : " + str(len(contours)))
    
    # Extract the bounding boxes from the YOLO results
    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")
    for index, row in px.iterrows():
        # Extract the class id and confidence score for the bounding box
        class_id = int(row[5])
        conf = row[4]

        # Find contours in the thresholded mask
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Check if the class is a person and the confidence score is above a threshold
        if conf > 0.3 and class_id == 5:
            # Extract the coordinates of the bounding box
            x = x1 = int(row[0])
            y = y1 = int(row[1])
            w = x2 = int(row[2])
            h = y2 = int(row[3])
            # Draw a rectangle around the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            mask1 = np.zeros_like(image)
            mask1[y:y+h, x:x+w] = 255
            # Extract the frame of the worker from the bounding box
            worker_frame = frame[y1:y2, x1:x2, :]
            image_frame = thresh[y:y+h, x:x+w]
            contours, _ = cv2.findContours(
                image_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.imshow('worker_frame', worker_frame)
            print(str(index)+" : " + str(len(contours)))

            cv2.drawContours(image_frame, contours, -1, (0, 255, 0), 3)
            # Run the YOLO model on the worker frame
            print(str(index)+"--->")
            print(co)
            
            try:
                if flag or not (is_closer(co.values[index], len(contours), 1000)) or count == 1 or count % (30*2) == 0:
                    worker_results = model.predict(worker_frame)
                    worker_results_dict[(x1, y1)] = worker_results
                    co[index] = len(contours)
                    flag = True
                # Store the results for later use
                else:
                    worker_results = worker_results_dict.get((x1, y1), None)
            except:
                print("exception")
                worker_results = model.predict(worker_frame)
                worker_results_dict[(x1, y1)] = worker_results
                co[index] = len(contours)
                flag = True
                #cv2.waitKey(10000)
                # Get the results from dictionary if available
            # Extract the bounding boxes from the worker YOLO results
            worker_a = worker_results[0].boxes.boxes
            worker_px = pd.DataFrame(worker_a).astype("float")
            helmet_wearing = False
            vest_wearing = False
            mask_wearing = False
            # Check if the confidence is above a certain threshold and the class is helmet or vest or mask
            for _, worker_row in worker_px.iterrows():
                label = ""
                worker_class_id = int(worker_row[5])
                worker_conf = worker_row[4]
                if worker_conf > 0.25 and worker_class_id in [0, 7]:
                    if worker_class_id == 0:
                        helmet_wearing = True
                    elif worker_class_id == 7:
                        vest_wearing = True
                #if worker_conf > 0.1 and worker_class_id in [1]:
                mask_wearing = True
            # Labeling the person accroding to detection
            label += str(index)

            if helmet_wearing and vest_wearing and mask_wearing:
                label += "yes"
                color = (0, 255, 0)
            else:
                label += "no "
                if not helmet_wearing:
                    label += "helmet "
                if not vest_wearing:
                    label += "vest "
                if not mask_wearing:
                    label += "mask"
                color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame, label, (x1, y1),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
            cv2.imshow('Contoursparts', image_frame)
            # cv2.waitKey(1000)

    # displaying the frame
        cv2.imshow("RGB", frame)
        #cv2.imshow('Contours', image)

    # check for escape key
        if cv2.waitKey(1) & 0xFF == 27:
            break
etime=time.time()

print("Elapsed time: {:.2f} seconds", format(etime-stime))
# releasing the video capture object and closing all windows
cap.release()
cv2.destroyAllWindows()

