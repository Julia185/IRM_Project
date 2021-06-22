# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 17:09:56 2021

@author: Julia
"""

import os
import cv2
import json
import numpy as np
import random


class DataGenerator :
    def __init__(self, path, classes=None, sequences=None,
                 sequences_path='patientsSequences.json',
                 locations_path='patientsLocations.json',
                 patient_error_path='patientsErrors.json',
                 output_path='./preload_data'):
        
        if sequences is None:
            sequences = ['STIR', 'T1']

        if classes is None:
            classes = ["RMI_OK", "RMI_INF", "RMI_DEG"]

        self.data_path = path

        self.sequences = sequences
        self.sequences_path = os.path.join(self.data_path, sequences_path)

        self.classes = classes
        self.nb_classes = len(self.classes)
        self.locations_path = os.path.join(self.data_path, locations_path)

        self.patients = {}
        self.patient_error_path = os.path.join(self.data_path, patient_error_path)
        self.output_path = output_path
    
    
    
    def generate_data(self,size):
        PatientsSequences = {}
        with open(self.sequences_path,'rb') as f:
            print("Load Patients Sequence json")
            text = json.load(f)

            # For each patient, we associate the MRI.
            for key,value in text.items():
                PatientsSequences[int(key)] = value

        PatientsLocations={}
        with open(self.locations_path,'rb') as f:
            print("Load Patients Location json")
            text=json.load(f)
            
            # For each patient, we locate the data.
            for key,value in text.items():
                PatientsLocations[int(key)]=value

        X = []
        Y = []
        classes_list = os.listdir(os.path.join(self.data_path, 'images\\raw'))

        for c in classes_list:
            file_list = os.listdir(os.path.join(os.path.join(self.data_path, 'images\\raw'), c))

            patient_set = set()         # Unordered unique set.
            for f in file_list:         # For each picture in one forlder.
                if '.png' in f:
                    pid = int(f[:4])    # pid = Patient ID.
                    patient_set.add(pid)
            self.patients[c] = patient_set   # Stores the name of the patients of one direct linked to direct name.

        PatientsOK = {}
        min_cuts = 100
        PatientsErrors = {}

        for c in classes_list:
            for pid in self.patients[c]:
                #print('Class:',c,' - Patient:',pid)
                m1 = PatientsSequences[pid]['STIR'][0][1]       # STIR photo number.
                if 'T1' in PatientsSequences[pid]:
                    m2 = PatientsSequences[pid]['T1'][0][1]     # T1 photo number.
                
                    if m1 != m2 :
                        l1 = PatientsLocations[pid]['T1'][0]
                        l2 = PatientsLocations[pid]['STIR'][0]
                        PatientsErrors[pid]={'CL':c,'T1':m1,'STIR':m2,'T1Cuts':l1,'STIRCuts':l2}
                        #print('Different length for',pid,'(',m1,'-',m2,')',l1,l2)
                        PatientsOK[pid] = 0
                    else:
                        PatientsOK[pid] = 1
                else:
                    PatientsOK[pid] = 0
                
                m = min(m1, m2)
                min_cuts = min(min_cuts, m)
                #print(pid,PatientsSequences[pid],m)
                
        f = open("PatientsErrors.json","w")
        for pid in PatientsErrors.keys():
            f.write('{}:{}\n'.format(pid,PatientsErrors[pid]))
        f.close()
  
        half_cut = int(min_cuts / 2)
        nb_frames = min_cuts * 2
  
        for c in classes_list:
            for pid in self.patients[c]:
                if pid not in PatientsOK:
                    continue
                if PatientsOK[pid]:
                    simage = np.zeros((size, size, nb_frames))
                    
                    i = 0
                    error = 0
                
                    # s = 'STIR' || s = 'T1'
                    for s in self.sequences: 
                        # On coupe les data en 2 paquets.
                        half_nb_imgs = int(PatientsSequences[pid][s][0][1]/2)
                        first_img_id = PatientsSequences[pid][s][0][0] + half_nb_imgs - half_cut
                    
                        for img_id in range(min_cuts):
                            cut_id = first_img_id + img_id
                            file = os.path.join(self.data_path, c) + '/' + str(pid).zfill(4) + '_' + s + '_' + str(cut_id).zfill(4) + '.png'
                            #print('Load',file)
                            
                            image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)                            
                            
                            try:
                                image = cv2.resize(image, (size, size))
                                simage[:, :, i] = image
                            except Exception as e:
                                #print(f'Error found at {file} & {cut_id}\n')
                                error += 1
                            
                            i += 1

                    X.append(simage)

                    y = [0]*len(self.classes)
                    y[self.classes.index(c)] = 1
                    Y.append(y)

        X = np.asarray(X)
        Y = np.asarray(Y)
    
        return X,Y,min_cuts
    
    
    
    def loadData(self,size,generate_array=True):
        '''
        Function to load the data.

        Parameters
        ----------
        size : int
            Size of the image.
        generate_array : Boolean, optional
            Variable used to see if we need to generate arrays. The default is True.

        Returns
        -------
        X : numpy array
            DESCRIPTION.
        Y : numpy array
            DESCRIPTION.
        test_x : numpy array.
            DESCRIPTION.
        test_y : numpy array
            DESCRIPTION.
        val_x : numpy array
            DESCRIPTION.
        val_y : numpy array
            DESCRIPTION.
        min_cuts : int
            DESCRIPTION.

        '''
        
        print("Loading Data...")
        
        # We generate the npy files for the different sizes.
        if generate_array: 
            X, Y, min_cuts = self.generate_data(size)
            print('MinCuts:',min_cuts)
            
            if not os.path.exists(self.output_path):
                print("NO")
                os.mkdir(self.output_path)

            with open(os.path.join('X_{}.npy'.format(size)), 'wb') as f:
                np.save(f,X)
            with open(os.path.join('Y_{}.npy'.format(size)), 'wb') as f:
                np.save(f,Y)
        else:
            with open(os.path.join('X_{}.npy'.format(size)), 'rb') as f:
                X=np.load(f)
            with open(os.path.join('Y_{}.npy'.format(size)), 'rb') as f:
                Y=np.load(f)
        
            min_cuts = int(X.shape[3] / 2)
                
        print("Loadind data Done !")

        X, Y, test_x, test_y, val_x, val_y = self.generate_test_data(X, Y, test_size=0.25, val_size=0.25)
        
        return X, Y, test_x, test_y, val_x, val_y, min_cuts
    
    
    
    def getErrorsOnHealthy(test_y,best_pred,idx_test):
        errors_on_healthy=[]
        for i in range(len(test_y)):
            if test_y[i]==0 and best_pred[i]!=test_y[i]:
                errors_on_healthy.append(idx_test[i])
        return errors_on_healthy
    
    
    
    def generate_test_data(self, X, Y, test_size=1/10, val_size=3/10):
        '''
        Function that generate the data for the test part of CNN.

        Parameters
        ----------
        X : numpy array
            DESCRIPTION.
        Y : numpy array
            DESCRIPTION.
        test_size : int, optional
            DESCRIPTION. The default is 1/10.
        val_size : int, optional
            DESCRIPTION. The default is 3/10.

        Returns
        -------
        X : numpy array
            DESCRIPTION.
        Y : numpy array
            DESCRIPTION.
        test_x : numpy array
            DESCRIPTION.
        test_y : numpy array
            DESCRIPTION.
        val_x : TYPE
            DESCRIPTION.
        val_y : TYPE
            DESCRIPTION.

        '''
        
        print("Generating Test data...")
        
        test_x = None
        test_y = None

        val_x = None
        val_y = None

        for i in range(0, self.nb_classes):
            _, classes = np.nonzero(Y)
            current_indices = np.where(classes == i)[0]            
            current_test_size = round(len(current_indices) * test_size)
            current_val_size = round(len(current_indices) * val_size)
            to_remove_number = current_test_size + current_val_size

            start_index = current_indices[0]
            new_data = X[start_index:start_index + to_remove_number]

            if test_x is None:
                test_x = new_data[0:current_test_size]
                val_x = new_data[current_test_size:current_val_size+current_test_size]
            else:
                test_x = np.concatenate((test_x, new_data[0:current_test_size]))
                val_x = np.concatenate((val_x, new_data[current_test_size:current_val_size+current_test_size]))

            X = np.delete(X, np.arange(start_index, start_index + to_remove_number, 1), axis=0)

            new_data = Y[start_index:start_index + to_remove_number]
            if test_y is None:
                test_y = new_data[0:current_test_size]
                val_y = new_data[current_test_size:current_val_size+current_test_size]
            else:
                test_y = np.concatenate((test_y, new_data[0:current_test_size]))
                val_y = np.concatenate((val_y, new_data[current_test_size:current_val_size+current_test_size]))

            Y = np.delete(Y, np.arange(start_index, start_index + to_remove_number, 1), axis=0)

        print("Generating test data Done !")
        return X, Y, test_x, test_y, val_x, val_y