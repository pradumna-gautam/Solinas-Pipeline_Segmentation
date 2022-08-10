# -*- coding: utf-8 -*-
"""
Created on Wed June 7 08:28:21 2021

@author: Pradumna
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import signal
import pywt
#import hfda
#import ewtpy
import pywt as py
from scipy.signal import hilbert, chirp, savgol_filter,find_peaks, peak_prominences
import statistics
#from PyEMD import EMD
import csv
import skinematics
import math
from time import sleep
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from skinematics.sensors.xio_ngimu import NGIMU
from skinematics.sensors.manual import MyOwnSensor
import argparse
from datetime import date
from scipy import constants
import sys
from scipy.signal import savgol_filter as sg
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
#import traja
os.listdir('/home/PRADUMNA/Desktop/IMU_collision_data')

path = '/home/PRADUMNA/Desktop/IMU_collision_data'
file = '/imu_26_oct_collisions_2.csv'
data_label = pd.read_csv(path + file, encoding= 'unicode_escape')
data_label

test_data1 = data_label.drop(columns= 'Unnamed: 0')
Visualizing the acceleration and gyroscope data
plt.plot(test_data1['accX'])
plt.plot(test_data1['accY'])

plt.xlabel('time [sec]')
plt.ylabel('Force [mg]')

plt.plot(test_data1['gyroX'])
plt.plot(test_data1['gyroY'])
plt.plot(test_data1['gyroZ'])
plt.title('angular velocity')

x = test_data1['accX']
y = test_data1['accY']
z = test_data1['accZ']
#Analyzing the moving variance of acc or gyro daat
vary_x = x.rolling(10).var()
#put any other data in place of x to see it's moving variance 
#but make sure it's a pandas series. The window of moving varinace can also be changes.
plt.plot(vary_x)
plt.ylabel('moving variance')

 
# converting acc and gyro data to numpy array
accX = np.array(test_data1['accX'])
accY = np.array(test_data1['accY'])
accZ = np.array(test_data1['accZ'])
gyroX = np.array(test_data1['gyroX'])
gyroY = np.array(test_data1['gyroY'])
gyroZ = np.array(test_data1['gyroZ'])
#Applying 1-D wavelet decomposition on the acc data
coeffs = py.wavedec(accX, 'haar', level=1)
cA1, cD1 = coeffs
# here we used haar wavelet, to find only sharper peaks we can take other wavelets but not required right now
# We used 1st level decomposition.
#analysing the 1st decompostion of wavelet
plt.plot(signal.decimate(accX,2))
plt.plot(cD1)
plt.title('wavelet 1 deco')

#Note the offset is revoed by wavelets and the collision is highlighed in the data(the data due to movment of imu is reduced near 0)
 
#analysing the 1st decompostion on gyroscope
coeffs = py.wavedec(gyroX, 'haar', level=1)
cA1, cD1 = coeffs
plt.plot(signal.decimate(gyroX,2))
plt.plot(cD1)
plt.title('wavelet 1 deco')

 
 
#puting the different dirrection of acc and gyro to one array. (this is for estimation oriantation)
acc = []
gyro = []
mag = []

acc.append(accX)
acc.append(accY)
acc.append(accZ)

gyro.append(gyroX)
gyro.append(gyroY)
gyro.append(gyroZ)
acc = np.array(acc)
acc = (acc)/100 
gyro = np.array(gyro)
gyro = (gyro*0.0174532925)/1000 # converting from degree to radians
time = np.linspace(0,len(accX)/25,len(accX))
# setting time array
initial_orientation = np.array([[1,0,0],
                                [0,0,-1],
                                [0,1,0]])


initialPosition = [0,0,0]

rate = 25
#below is a code to find the orientation and trajectory from acc and gyro data using the skinematics package
pos1 = skinematics.imus.analytical(initial_orientation, np.transpose(gyro), initialPosition, np.transpose(acc), rate)
#visualizing the quaternion
plt.plot(pos1[0][:,1:4])
plt.legend(["X-axis", "Y-axis","Z-axis"], loc ="lower right")

coeffs = py.wavedec(pos1[0][:,1], 'haar', level=1)
cA1, cD1 = coeffs
plt.plot(signal.decimate(pos1[0][:,1],2))

plt.plot(cD1)
plt.title('wavelet 1 deco')
 
#visualizing the oriantion in 3-D map
fig = plt.figure()
ax = Axes3D(fig)
numDataPoints = len(time)
#dataSet = np.array([x_axis, y_axis, z_axis])
# NOTE: Can't pass empty arrays into 3d version of plot()
line = plt.plot(pos1[0][:,1], pos1[0][:,2],pos1[0][:,3], lw=2, c='g')[0] # For line plot
 
# AXES PROPERTIES]
# ax.set_xlim3d([limit0, limit1])
ax.set_xlabel('Time [sec]')
ax.set_ylabel('X [m]')
ax.set_zlabel('Y [m]')
ax.set_title('Orientation of Iglobus')

plt.show()

#visualizing the oriantion in 3-D plot, (Note: This may not be accurate at all)
fig = plt.figure()
ax = Axes3D(fig)
numDataPoints = len(time)
#dataSet = np.array([x_axis, y_axis, z_axis])
# NOTE: Can't pass empty arrays into 3d version of plot()
line = plt.plot(time, pos1[1][:,0]/10, pos1[1][:,1]/10, lw=2, c='g')[0] # For line plot
 
# AXES PROPERTIES]
# ax.set_xlim3d([limit0, limit1])
ax.set_xlabel('Time [sec]')
ax.set_ylabel('X [m]')
ax.set_zlabel('Y [m]')
ax.set_title('Trajectory of Iglobus')


 
