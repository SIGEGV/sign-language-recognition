import cv2
import os
import uuid
import time

# databse path 
DB_PATH=f"database"

#createing labels for database; 
LABELS=['hello', 'yes']
number_of_img=20 #20 images per label 

for label in LABELS:
        directory = os.path.join(DB_PATH, label)
        if not os.path.exists(directory):
            os.makedirs(directory)
        capture=cv2.VideoCapture(0)
        print("capturing images for the label:"+label)
        time.sleep(5)
        for imgnum in range(number_of_img):
            ret,frame=capture.read()
            imagename=os.path.join(DB_PATH, label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
            cv2.imwrite(imagename,frame)
            cv2.imshow('Frame', frame)
            time.sleep(2)
            
            if cv2.waitKey(1) & 0xFF ==ord('q'):
                break
        capture.release()