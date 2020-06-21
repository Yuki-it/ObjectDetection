# OpenCV Python program to detect cars in video frame 
# import libraries of python OpenCV  
import cv2 
import os
import time
  
# capture frames from a video 
cap = cv2.VideoCapture('car_person.mp4') 
  
# Trained XML classifiers describes some features of some object we want to detect 

common_path = 'C:\\Users\\yukir\\anaconda3\\envs\\opencv\\Lib\\site-packages\\cv2\\data'

person_cascade = cv2.CascadeClassifier(os.path.join(car_path,'haarcascade_fullbody.xml')) 
car_cascade = cv2.CascadeClassifier(os.path.join(car_path,'cars.xml'))
  
# loop runs if capturing has been initialized. 
while True: 
    # reads frames from a video 
    ret, frames = cap.read() 
      
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY) 
      
  
    # Detects cars of different sizes in the input image 
    people = car_cascade.detectMultiScale(gray, 1.1, 1) 
    cars = car_cascade.detectMultiScale(gray, 1.1, 1) 
      
    # To draw a rectangle in each cars 
    for (x,y,w,h) in cars: 
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2) 
  
    
    for (x,y,w,h) in people: 
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2) 

    # Display frames in a window  
    cv2.imshow('video2', frames) 
      
    # Wait for Esc key to stop 
    if cv2.waitKey(1) == ord('q'): 
        break
  
# De-allocate any associated memory usage 
cap.release()
cv2.destroyAllWindows() 
