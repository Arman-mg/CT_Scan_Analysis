# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 18:59:31 2021

"""

import numpy   as np
import nibabel as nib # to read NII files
import matplotlib.pyplot as plt
from sub.CTscan import CTscan
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from itertools import product

    
#%% main part

plt.close('all')    
fold1='./data/ct_scans'
fold2='./data/lung_mask'
fold3='./data/infection_mask'
fold4='./data/lung_and_infection_mask'
f1='/coronacases_org_001.nii'
f2='/coronacases_001.nii'



#%% Examine one slice of a ct scan and its annotations
index=132
filepath=fold1+f1+f1
a=CTscan(index)
sample_ct=a.read_nii(filepath)
a.hist()
Ncluster=5 #number of desired cluster
ifind=1# second darkest color
D=a.Kmeans(Ncluster,ifind)#Use Kmeans to perform color quantization of the image
eps=2
min_samples=5
a.find_lungs(eps,min_samples)#DBSCAN to find the lungs in the image
a.quantized()#generate a new image with the two darkest colors of the color-quantized image
a.final_lung_masks() #Final lung masks
a.find_GGO()#Find ground glass opacities
L_meas,R_meas=a.infection_meas()#ifection measurement for each lung
print('Left lung infection severity :')
print(L_meas)
print('Right lung infection severity :')
print(R_meas) 


