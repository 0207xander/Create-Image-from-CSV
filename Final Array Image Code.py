#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.style.use('fivethirtyeight')
import csv
import math
from csv import reader
import pandas as pd
import uproot


# In[31]:


signal=uproot.open(r"D:\new_TTHH.root")
tree=signal["OutputTree"]
branches = tree.arrays()
numevents=len(branches["met"])            


def createImage(event):

    start_time = time.time()            
 
    # Jet part

    # loop over events
    for i in range(numevents):

        # a way to only generate one event for now
        if i == int(event):              

            # creating the jet arrays    

            eta_values = []
            phi_values = []
            pT_values = []
            b_values = []

            for j in range(1, 11):
                pT_values.append(branches['jet' + str(j) + 'pT'][event])
            for j in range(1, 11):
                eta_values.append(branches['jet' + str(j) + 'eta'][event])
            for j in range(1, 11):
                phi_values.append(branches['jet' + str(j) + 'phi'][event])
            for j in range(1, 11):
                b_values.append(branches['jet' + str(j) + 'btag'][event])           # Fill b_values and pT arrays in same loop

            # creating the lepton arrays

            lepton_eta_values = []
            lepton_phi_values = []
            lepton_pT_values = []

            for j in range(1, 4):
                lepton_pT_values.append(branches['lepton' + str(j) + 'pT'][event])
            for j in range(1, 4):
                lepton_eta_values.append(branches['lepton' + str(j) + 'eta'][event])
            for j in range(1, 4):
                lepton_phi_values.append(branches['lepton' + str(j) + 'phi'][event])     

        else:
            pass

        # Turning the jet values into arrays

    points = None
    points = list(zip(eta_values, phi_values, pT_values))
    
    #pT_values = np.log10(pT_values)

    eta_array = np.array(eta_values)
    phi_array = np.array(phi_values)
    pT_array = np.array(pT_values)
    b_array = np.array(b_values)

        # CODE TO CREATE THE 2D ARRAY 

    eta_bin_size = 0.2
    phi_bin_size = math.pi/20
    eta_index = np.round(eta_array / eta_bin_size).astype(int)
    phi_index = np.round(phi_array / phi_bin_size).astype(int)
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
            larray[ie].append([0,0,0])


          # Appending the pT values to the array of zeros

    for i in range(0, len(pT_values)):
        if pT_values[i] > 0:
            array[eta_index[i] + int(num_eta_bins / 2), phi_index[i] + int(num_phi_bins / 2)] = pT_values[i]             
            larray[eta_index[i] + int(num_eta_bins / 2)][phi_index[i] + int(num_phi_bins / 2)] = [pT_values[i],0,0]

    pTs = []

    max_jet_value = max(pT_values)
    
    for i in range(0, len(pT_values)):
        if pT_values[i] > 0:
            pT_scaled = pT_values[i] / max_jet_value
            array[eta_index[i] + int(num_eta_bins / 2), phi_index[i] + int(num_phi_bins / 2)] = pT_values[i]
            if b_values[i] != 1:
                larray[eta_index[i] + int(num_eta_bins / 2)][phi_index[i] + int(num_phi_bins / 2)] = [pT_scaled,0,0]
            else:
                larray[eta_index[i] + int(num_eta_bins / 2)][phi_index[i] + int(num_phi_bins / 2)] = [0,pT_scaled,0]


        # Lepton part                              

        # Turning the lepton values from above into arrays

    points = None
    points = list(zip(lepton_eta_values, lepton_phi_values, lepton_pT_values))

    lepton_eta_array = np.array(lepton_eta_values)
    lepton_phi_array = np.array(lepton_phi_values)
    lepton_pT_array = np.array(lepton_pT_values)

        # CODE TO CREATE THE 2D HISTOGRAM FOR LEPTONS

    eta_index = np.round(lepton_eta_array / eta_bin_size).astype(int)
    phi_index = np.round(lepton_phi_array / phi_bin_size).astype(int)

    array = np.zeros(shape = (num_eta_bins,num_phi_bins))
    np.set_printoptions(threshold=np.inf)
    lepton_larray=[]
    for ie in range(num_eta_bins):
        lepton_larray.append([])
        for ip in range(num_phi_bins):
            lepton_larray[ie].append([0,0,0])

    for i in range(0, len(lepton_pT_values)):
        if lepton_pT_values[i] > 0:
            array[eta_index[i] + int(num_eta_bins / 2), phi_index[i] + int(num_phi_bins / 2)] = lepton_pT_values[i]             
            lepton_larray[eta_index[i] + int(num_eta_bins / 2)][phi_index[i] + int(num_phi_bins / 2)] = [lepton_pT_values[i],0,0]

    max_lepton_value = 0
    for j in lepton_larray:
        for i in j:
            if i[0] > max_lepton_value:
                max_lepton_value = i[0]

    for i in range(0, 50):
        for j in range(0, 40):
            if lepton_larray[i][j][0] != 0:
                lepton_larray[i][j][0] = lepton_larray[i][j][0] / max_lepton_value   

    for i in range(len(lepton_larray)):
        for j in range(len(lepton_larray[i])):
            if lepton_larray[i][j][0] != 0:
                lep_val = lepton_larray[i][j][0]
                #JOIN THE LEPTON VALUES TO THE JET ARRAY TO COMBINE BOTH ARRAYS
                larray[i][j][2] = lep_val
                
#     for i in range(0, 50):            ----> turns background white, changes point color
#         for j in range(0, 40):
#             for k in range(0, 3):
#                 if larray[i][j] == [0, 0, 0]:
#                     larray[i][j] = [1, 1, 1]           

    img = plt.imshow(larray)

    
createImage(0)    
