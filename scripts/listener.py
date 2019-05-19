#!/usr/bin/env python3
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import sys
import cv2

# Instantiate CvBridge
bridge = CvBridge()
import glob
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
import numpy as np

# from keras.models import Sequential, load_model
# from keras.preprocessing.image import ImageDataGenerator
# from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Lambda, Cropping2D, ELU
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers.convolutional import Convolution2D
# from keras import regularizers, optimizers, initializers
# from keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard, Callback
print("Done importing data")

import subprocess
import execnet
import cv2
from cv_bridge import CvBridge, CvBridgeError
def call_python_version(Version, Module, Function, ArgumentList):
    gw      = execnet.makegateway("popen//python=python%s" % Version)
    channel = gw.remote_exec("""
        from %s import %s as the_function
        channel.send(the_function(*channel.receive()))
    """ % (Module, Function))
    channel.send(ArgumentList)
    return channel.receive()


def image_callback(img):
    print("1")
    val=np.frombuffer(img)
    # result = call_python_version("2.7", "my_module", "my_function",[img]) 
    print(result)
    # img = CvBridge().imgmsg_to_cv2(img,desired_encoding="bgr8")
    # print(img.shape)
    # print("Received an image!")
    # image = img[50:140, 0:320]
    # print(image.shape)                
    # # Resize to 200x66 pixel
    # image1 = cv2.resize(image, (200,66), interpolation=cv2.INTER_AREA)
    # print(image1.shape)
    # images=np.reshape(image1, (1, 66,200,3))
        
    # image.reshape()
    # val=model.predict(images)
    # print(val)
    # #np_arr = np.fromstring(msg, np.uint8) 
    # #np_img = np_arr.reshape((480, 640, 3)) 
    # print("0")
    # #image=image[50:140, 0:320]
    # print("1")
    # # Resize to 200x66 pixel
    # image = cv2.resize(msg, (200,66), interpolation=cv2.INTER_AREA)
    # print("2")
    # images=np.reshape(image, (1, 66,200,3))    
    # # image.reshape()
    # val=model.predict(images)
    # print("3")
    # print(val)

def main():
    rospy.init_node('image_listener')
    print('I am listening')
    import sys
    print(sys.version)
    # Define your image topic
    image_topic = "/synchronized_image_raw"
    # model=load_model('/home/akashbaskaran/CNN/model-e007.h5')

    # print(model.summary())

    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()




# image = cv2.imread('/home/akashbaskaran/CNN/frame000225.jpg')
# a=1

# result = call_python_version("2.7", "my_module", "my_function",[]) 
# # print(result) 
# # result = call_python_version("2.7", "my_module", "my_function",  
# #                              ["Mrs", "Wolf"]) 
# print(result)
# # a=5
# b=6
# python3_command = "/home/akashbaskaran/catkin_ws/src/predictor/scripts/test.py a b"  # launch your python2 script using bash

# process = subprocess.Popen(python3_command.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()  # receive output from the python2 script