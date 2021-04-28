# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:12:17 2020

@author: LAURI
"""


import time

#import keras
#from keras.preprocessing.image import ImageDataGenerator
#from keras import optimizers
#from keras.models import Sequential, Model 
#from keras.layers import *
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
#from keras.wrappers.scikit_learn import KerasClassifier
#
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score
#from sklearn.metrics import f1_score
#from sklearn.metrics import cohen_kappa_score
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import classification_report

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import keras.metrics as km

import tensorflow as tf
from keras import backend as K

import json


DATA_DIR = r"C:\Users\julia\Desktop\UTBM\Cours\Branche\INFO4\TX52\images\raw"

CLASSES = ["RMI_OK", "RMI_INF", "RMI_DEG"]
SEQUENCES=['STIR','T1']

Patients = dict()

nb_classes = len(CLASSES)


num_cores = 96
CPU = False

if not CPU:
    num_GPU = 1
    num_CPU = 1
else:
    num_CPU = 1
    num_GPU = 0


#config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
#                        inter_op_parallelism_threads=num_cores, 
#                        allow_soft_placement=True,
#                        device_count = {'CPU' : num_CPU,
#                                        'GPU' : num_GPU}
#                       )


#session = tf.Session(config=config)
#K.set_session(session)


def generate_data(input_dir,size):
    PatientsSequences = {}
    with open('PatientsSequences.json','rb') as f:
        text = json.load(f)

        # For each patient, we associate the MRI.
        for key,value in text.items():
            PatientsSequences[int(key)] = value

    PatientsLocations={}
    with open('PatientsLocations.json','rb') as f:
        text=json.load(f)

        # For each patient, we locate the data.
        for key,value in text.items():
            PatientsLocations[int(key)]=value

    X = []
    Y = []
    classes_list = os.listdir(input_dir)

    for c in classes_list:
        file_list = os.listdir(os.path.join(input_dir, c))

        patient_set = set()         # Unordered unique set.
        for f in file_list:         # For each picture in one forlder.
            if '.png' in f:
                pid = int(f[:4])    # pid = Patient ID.
                patient_set.add(pid)
        Patients[c] = patient_set   # Stores the name of the patients of one direct linked to direct name.

    PatientsOK = {}
    min_cuts = 100
    PatientsErrors = {}

    for c in classes_list:
        for pid in Patients[c]:
            print('Class:',c,' - Patient:',pid)
            m1 = PatientsSequences[pid]['STIR'][0][1]       # STIR photo number.
            if 'T1' in PatientsSequences[pid]:
                m2 = PatientsSequences[pid]['T1'][0][1]     # T1 photo number.
                
                if m1 != m2 :
                    l1 = PatientsLocations[pid]['T1'][0]
                    l2 = PatientsLocations[pid]['STIR'][0]
                    PatientsErrors[pid]={'CL':c,'T1':m1,'STIR':m2,'T1Cuts':l1,'STIRCuts':l2}
                    print('Different length for',pid,'(',m1,'-',m2,')',l1,l2)
                    PatientsOK[pid] = 0
                else:
                    PatientsOK[pid] = 1
            else:
                PatientsOK[pid] = 0
            m = min(m1, m2)
            min_cuts = min(min_cuts, m)
#           print(pid,PatientsSequences[pid],m)

    # On crée un fichier json pour mettre les mauvais échantillons ?
    f = open("PatientsErrors.json","w")
    for pid in PatientsErrors.keys():
        f.write('{}:{}\n'.format(pid,PatientsErrors[pid]))
    f.close()

    half_cut = int(min_cuts/2)

    nb_frames = min_cuts*2
    for c in classes_list:
        for pid in Patients[c]:
            if PatientsOK[pid]:
                simage = np.zeros((size,size,nb_frames))
                i = 0
                
                # s = 'STIR' || s = 'T1'
                for s in SEQUENCES: 
                    # On coupe les data en 2 paquets.
                    half_nb_imgs = int(PatientsSequences[pid][s][0][1]/2)
                    first_img_id = PatientsSequences[pid][s][0][0] + half_nb_imgs - half_cut
                    
                    for img_id in range(min_cuts):
                        cut_id = first_img_id + img_id
                        file = os.path.join(input_dir, c)+'/'+str(pid).zfill(4)+'_'+s+'_'+str(cut_id).zfill(4)+'.png'
#                        print('Load',file)
                        image = cv2.imread(file,0)
                        height, width = image.shape[:2]
                        image = cv2.resize(image, (size, size))
                        simage[:,:,i] = image
                        i += 1

                X.append(simage)

                y = [0]*len(CLASSES)
                y[CLASSES.index(c)] = 1
                Y.append(y)

    X = np.asarray(X)
    Y = np.asarray(Y)
    
    return X,Y,min_cuts


def loadData(size,generate_array=True):
    if generate_array:
        X, Y, min_cuts = generate_data(DATA_DIR,size)
        print('MinCuts:',min_cuts)

        with open('X_{}.npy'.format(size),'wb') as f:
            np.save(f,X)
        with open('Y_{}.npy'.format(size),'wb') as f:
            np.save(f,Y)
    else:
        with open('X_{}.npy'.format(size),'rb') as f:
            X=np.load(f)
        with open('Y_{}.npy'.format(size),'rb') as f:
            Y=np.load(f)
        
        min_cuts=int(X.shape[3]/2)

    return X,Y,min_cuts


def getErrorsOnHealthy(y_test,best_pred,idx_test):
    errors_on_healthy=[]
    for i in range(len(y_test)):
        if y_test[i]==0 and best_pred[i]!=y_test[i]:
            errors_on_healthy.append(idx_test[i])
    return errors_on_healthy


# Generate files speeding the subsequent loading of data
for size in [32,64,128,256,512]:
    print('Generate data for images of size',size)
    loadData(size,True)


# Load 128x128 images
size=128

X,Y,seq_len=loadData(size,False)

input_shape=(size,size,seq_len*2)
print(input_shape)
