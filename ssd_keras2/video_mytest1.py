import os
import keras
import pickle
from video_mytest import VideoTest

import sys
sys.path.append("..")
from ssd import SSD300 as SSD

input_shape = (512,512,3)

# Change this if you run with other classes than VOC
class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"];
#class_names = [ "bicycle", "bus", "car",  "dog", "horse", "motorbike", "person",  "train"];
NUM_CLASSES = len(class_names)

model = SSD(input_shape, num_classes=NUM_CLASSES)

# Change this path if you want to use your own trained weights
model.load_weights('../weights_SSD300.hdf5') 
        
vid_test = VideoTest(class_names, model, input_shape)

# To test on webcam 0, remove the parameter (or change it to another number
# to test on that webcam)
mainPATH    = 'C:\\Users\\yukir\\workspace\\opencv_work'
subMainPATH = os.path.join(mainPATH,'Vehicle-And-Pedestrian-Detection-Using-Haar-Cascades\\Main Project\\Main Project')

car_person  = os.path.join(mainPATH,'car_person.mp4')
pedestrians = os.path.join(subMainPATH,'Pedestrian Detection\\pedestrians.avi')
carVideo1   = os.path.join(subMainPATH,'Car Detection\\video.avi')
carVideo2   = os.path.join(subMainPATH,'Car Detection\\video1.avi')
BusVideo    = os.path.join(subMainPATH,'Bus Detection\\bus1.mp4')
BikeVideo   = os.path.join(subMainPATH,'Bike Detection\\two_wheeler2.mp4')

vid_test.run(car_person,16000,0.5)
