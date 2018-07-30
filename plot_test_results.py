#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 20:25:02 2018

@author: isrcig
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
#
#solid   -
#dashed --
#dotted  :

# t, episodio, elem_episodio, max_elem_episodio

folder_path = 'results/teste/'

data_path1 = 'results/teste/Goal1Vrob1_0-position_state_data.csv'
with open(data_path1, 'r') as f_position_state_data1:
    reader_position_state_data1 = csv.reader(f_position_state_data1, delimiter=',')
    # get header from first row
    headers_position_state_data1 = next(reader_position_state_data1)
    # get all the rows as a list
    position_state_data1 = list(reader_position_state_data1)
    # transform data into numpy array
    position_state_data1 = np.array(position_state_data1).astype(float)
    
print(headers_position_state_data1)
#print(position_state_data1.shape)
#print(data[:3])

# Plot the data
plt.plot(position_state_data1[:, 0],position_state_data1[:, 1])
#plt.plot(data_position_state_data[0, 0],data_position_state_data[0, 1], color='black', marker='x')
plt.plot(-1.7,-1.925, color='red', marker='x')
plt.plot(1.65, 1.625, color='red', marker='o')
plt.xlabel('x robot position')
plt.ylabel('y robot position')
#plt.show()

plt.savefig(folder_path + 'trajetoria.png')#, bbox_inches='tight')

plt.clf()
plt.plot(position_state_data1[:, 2], marker='x')
plt.xlabel('Time step')
plt.ylabel('distance to goal')
#plt.show()

plt.savefig(folder_path + 'distance_to_goal.png')

plt.clf()
plt.plot(position_state_data1[:, 3], marker='x')
plt.xlabel('Time step')
plt.ylabel('sum of distance to obstacle')
#plt.show()

plt.savefig(folder_path + 'sum_distance_to_obs.png')

plt.clf()
plt.plot(position_state_data1[:, 4], marker='x')
plt.xlabel('Time step')
plt.ylabel('Angle to goal')
#plt.show()

plt.savefig(folder_path + 'angle_to_goal.png')

