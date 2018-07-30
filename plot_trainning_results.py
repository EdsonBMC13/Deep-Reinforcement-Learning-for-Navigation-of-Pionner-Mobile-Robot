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

folder_path = 'results/treinamento/'

learn_path = 'results/treinamento/learn_data-200-200-100-50000_2.csv'
with open(learn_path, 'r') as f_learn:
    reader_learn = csv.reader(f_learn, delimiter=',')
    # get header from first row
    headers_learn = next(reader_learn)
    # get all the rows as a list
    data_learn = list(reader_learn)
    # transform data into numpy array
    data_learn = np.array(data_learn).astype(float)
    
print(headers_learn)
print(data_learn.shape)
#print(data[:3])

# Plot the data
plt.scatter(data_learn[:, 0],data_learn[:, 1], marker='o')
plt.xlabel('Time')
plt.ylabel('Episode')
#plt.show()

plt.savefig(folder_path + 'timeXepisode.png')#, bbox_inches='tight')

plt.clf()
plt.scatter(data_learn[:, 1],data_learn[:, 2])
plt.xlabel('Episode')
plt.ylabel('Number of elements')
#plt.show()

plt.savefig(folder_path + 'episodeXnumber_of_elements.png')

plt.clf()
plt.scatter(data_learn[:, 1],data_learn[:, 3])
plt.xlabel('Episode')
plt.ylabel('Max. number of elements')
#plt.show()

plt.savefig(folder_path + 'episodeXmax_number_of_elements.png')

loss_path = 'results/treinamento/loss_data-200-200-100-50000_2.csv'
with open(loss_path, 'r') as f_loss:
    reader_loss = csv.reader(f_loss, delimiter=',')
    # get header from first row
    headers_loss = next(reader_loss)
    # get all the rows as a list
    data_loss = list(reader_loss)
    # transform data into numpy array
    data_loss = np.array(data_loss).astype(float)
    
print(headers_loss)
print(data_loss.shape)
#print(data[:3])

plt.clf()
# Plot the data
plt.plot(data_loss[:, 0])
plt.xlabel('Time')
plt.ylabel('Loss')
#plt.show()

plt.savefig(folder_path + 'loss_training.png')

# linestyle='dashed', linewidth=5.0
#, marker='o', markersize=15.0, markeredgewidth=5.0,
