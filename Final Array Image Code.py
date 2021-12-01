#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.style.use('fivethirtyeight')
import csv
import math
import tkinter
from csv import reader
import pandas as pd
import matplotlib.colors
import matplotlib.cm as cm

color_max_pT=1000000.
color_min_pT=30000.

import uproot
signal=uproot.open(r"D:\new_TTHH.root")
tree=signal["OutputTree"]
branches = tree.arrays()
numevents=len(branches["met"])

eta_values_all = []
phi_values_all = []
pT_values_all = []
  
for j in range(1, 11):
    pT_values_all.append(branches['jet' + str(j) + 'pT'])
    eta_values_all.append(branches['jet' + str(j) + 'eta'])
    phi_values_all.append(branches['jet' + str(j) + 'phi'])  


def makePlotFromEvent(eventnumber):
    pT_values = [pT_values_all[k][eventnumber] for k in range(len(pT_values_all))]
    eta_values= [eta_values_all[k][eventnumber] for k in range(len(eta_values_all))]
    phi_values= [phi_values_all[k][eventnumber] for k in range(len(phi_values_all))]

    points = None
    points = list(zip(eta_values, phi_values, pT_values))

    eta_array = np.array(eta_values)
    phi_array = np.array(phi_values)
    pT_array = np.array(pT_values)

    # CODE TO CREATE THE 2D HISTOGRAM

    eta_bin_size = 0.2
    phi_bin_size = math.pi/20
    eta_index = np.floor(eta_array / eta_bin_size).astype(int)
    phi_index = np.floor(phi_array / phi_bin_size).astype(int)
    eta_min, eta_max = np.min(-5), np.max(5)
    phi_min, phi_max = np.min(-math.pi), np.max(math.pi)
    num_eta_bins = int(2*eta_max/eta_bin_size)  #50
    num_phi_bins = int(2*phi_max/phi_bin_size)  #40

    array = np.zeros(shape = (num_eta_bins,num_phi_bins))
    np.set_printoptions(threshold=np.inf)
    larray=[]
    for ie in range(num_eta_bins):
        larray.append([])
        for ip in range(num_phi_bins):
            larray[ie].append([1.,1.,1.])
    
    for i in range(0, len(pT_values)):
        if pT_values[i] > 0:
            #ptcolor_rgb=cm.Blues(min(math.log10(pT_values[i]/1000.-color_min_pT/1000.)/math.log10(color_max_pT/1000.),1.))[:3]
            ptcolor_rgb=cm.Blues(min(pT_values[i]/color_max_pT,1.))[:3]
            #print(ptcolor_rgb)
            array[eta_index[i] + int(num_eta_bins / 2), phi_index[i] + int(num_phi_bins / 2)] = pT_values[i]             
            larray[eta_index[i] + int(num_eta_bins / 2)][phi_index[i] + int(num_phi_bins / 2)] = ptcolor_rgb
            #print(pT_values[i])

    x,y = array.nonzero() #get the notzero indices
    if False:
        plt.scatter(x,y,c=array[x,y],s=50,cmap='winter',marker='s') #adjust the size to your needs
        plt.xlim(0, 50)
        plt.ylim(0, 40)
        plt.colorbar()
        plt.show()
    return larray


# loop over events
larray=None
for i in range(numevents):
    
    # just a way to only generate one event
    if i>0: break
     
    larray=makePlotFromEvent(i)
   
#Creating the signal events

x_train = []

iter = 10000
for i in range(0, iter):
    x_train.append(makePlotFromEvent(i))

rows, cols = (iter, 1)
y_train=np.array([[1] for i in range(iter)])

#Now to create the background events from a different root file

background=uproot.open(r"D:\new_TTBB.root")
tree=background["OutputTree"]
branches = tree.arrays()

for i in range(0, iter):
    x_train.append(makePlotFromEvent(i))

rows, cols = (iter, 1)
y_train2=np.array([[0] for i in range(iter)]) # append, not replace
y_train = np.concatenate((y_train, y_train2))

x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)

#Create the models architecture

model = Sequential()

#Add the first layer
model.add( Conv2D(32, (5,5), activation='relu', input_shape=(50,40,3)) )

#Add a pooling layer
model.add(MaxPooling2D(pool_size = (2,2)))

#Add another convolutional layer
model.add( Conv2D(32, (5,5), activation='relu') )

#Add another pooling layer
model.add(MaxPooling2D(pool_size = (2,2)))

#Add a flattening layer
model.add(Flatten())

#Add a layer with 1000 neurons
model.add(Dense(1000, activation='relu'))

#Add a dropout layer
model.add(Dropout(0.5))

#Add a layer with 500 neurons
model.add(Dense(500, activation='relu'))

#Add a dropout layer
model.add(Dropout(0.5))

#Add a layer with 250 neurons
model.add(Dense(250, activation='relu'))

#Add a layer with 10 neurons
model.add(Dense(1, activation='softmax'))

#Compile the model

model.compile(loss = 'categorical_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])

# Train the model with our data

model.summary()

hist = model.fit(x_train, y_train,
                batch_size = 100,
                epochs = 10,
                validation_split = 0.2)
