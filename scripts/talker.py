#! /usr/bin/python
# Copyright (c) 2015, Rethink Robotics, Inc.

# Using this CvBridge Tutorial for converting
# ROS images to OpenCV2 images
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

# Using this OpenCV2 tutorial for saving Images:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2

#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

#import cv2

# Instantiate CvBridge
bridge = CvBridge()
import glob
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
import numpy as np

from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Lambda, Cropping2D, ELU
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras import regularizers, optimizers, initializers
from keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard, Callback
print("Done importing data")


def image_callback(msg):
    image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
    print("Received an image!")
    #image = imgmsg_to_cv2(msg)
    #np_arr = np.fromstring(msg, np.uint8) 
    #np_img = np_arr.reshape((480, 640, 3)) 
    print("0")
    #image=image[50:140, 0:320]
    print("1")
    # Resize to 200x66 pixel
    image = cv2.resize(msg, (200,66), interpolation=cv2.INTER_AREA)
    print("2")
    images=np.reshape(image, (1, 66,200,3))    
    # image.reshape()
    val=model.predict(images)
    print("3")
    print(val)

def main():
    rospy.init_node('image_listener')
    print('I am listening')
    import sys
    print(sys.version)
    # Define your image topic
    image_topic = "/synchronized_image_raw"
    model=load_model('/home/akashbaskaran/CNN/model-e007.h5')

    print(model.summary())

    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()
