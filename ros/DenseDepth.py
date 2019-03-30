import os
import glob
import argparse
import depth2cloud
import cv2
import numpy as np
import subprocess
import time

import rospy
import math
import sys

from sensor_msgs.msg import PointCloud2
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2

# Kerasa / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from matplotlib import pyplot as plt

import pcl_util

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import layers
import loss
import utils



# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='../model_data/nyu.h5', type=str, help='Trained Keras model file.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': layers.BilinearUpSampling2D, 'depth_loss_function': loss.depth_loss_function}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

roscore = subprocess.Popen(['roscore'])
time.sleep(1)
try:
  rviz = subprocess.Popen(['rviz', '{}/DenseDepth.rviz'.format(os.getcwd())])
except:
  print('{}/DenseDepth.rviz'.format(os.getcwd()))
  pass

cnt = 0

cap = cv2.VideoCapture(0)

ret, img = cap.read()

scale_x=320.0/640.0
scale_y=240.0/480.0

# instrinsics =  [[ 810.56651485*scale_x,       0.0,           355.93294044*scale_x],
#                  [   0.0,     815.07382363*scale_y,          262.91874826*scale_y],
#                  [   0.0,                      0.0,          1.0]]

# Laptop
instrinsics =  [[ 738.7551115*scale_x,       0.0,           334.36906597*scale_x],
                [   0.0,     737.08804151*scale_y,          192.90978847*scale_y],
                [   0.0,                      0.0,          1.0]]

'''
Sample code to publish a pcl2 with python
'''
rospy.init_node('pcl2_pub')
pcl_pub = rospy.Publisher("/pcl_topic", PointCloud2)
rospy.loginfo("Initializing sample pcl2 publisher node...")

print(instrinsics)

K_Matrix_Inv = np.linalg.inv(instrinsics).astype(np.float32)
K_Matrix_Inv = K_Matrix_Inv[np.newaxis,:,:]

while ret:
  img_rgb = img[...,::-1]
  img = img/255.0
  img = np.expand_dims(img, axis=0)

  # Compute results
  outputs = utils.predict(model, img)

  # print('predict shape {}'.format(outputs.shape)) 

  cloud = depth2cloud.get_cloud(outputs, K_Matrix_Inv)
  cloud = np.squeeze(cloud)
  img_rgb = cv2.resize(img_rgb, (img_rgb.shape[1]//2,img_rgb.shape[0]//2))
  ros_msg = pcl_util.xyzrgb_array_to_pointcloud2(cloud, img_rgb, stamp=rospy.Time.now(), frame_id='map')

  #publish point cloud
  pcl_pub.publish(ros_msg)

  # Display results
  viz = utils.display_images(outputs.copy(), img.copy())

  cv2.imshow('test',viz)
  key = cv2.waitKey(10)
  if key == 27 or key == ord('q'):
    break

  cnt += 1
  ret, img = cap.read()

print('ctrl+c to stop')

roscore.wait()
rviz.wait()