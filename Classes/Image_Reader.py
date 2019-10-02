import numpy as np
import pandas as pd
import os
import cv2
import csv

class image_Reader():
    
    def __init__(self, image_Path, width, height, csv_path, classes) :
        
        self.image_path = image_Path
        self.classes = classes
        self.create_Directory()
        self.images = self.create_Directory()
        self.height = height
        self.width = width
        self.csv = csv_path
        self.data_Read()
        self.data = np.array(self.data_Read())
        self.X = np.ones((len(self.images), self.width, self.height), dtype = np.float32)
        self.X_norm = np.ones((len(self.images), self.width, self.height), dtype = np.float32)
        self.y = np.zeros([len(self.data), self.classes], dtype = int) 
        self.binary_digits = 20
        self.y_bin = np.zeros([len(self.data), self.binary_digits], dtype = int)
       
    # Create string array containing every file name and path    
    def create_Directory(self) :
        image_path_str = self.image_path + '{}'
        return [image_path_str.format(i) for i in os.listdir(self.image_path)]
    
    # Hold images in array
    def store_Image_Array(self):
        i = 0
        for image in self.images :
            self.X[i, :,:] = cv2.resize(cv2.imread(image, cv2.IMREAD_GRAYSCALE), (self.width, self.height),
                                    interpolation = cv2.INTER_CUBIC)
            i +=1
        return self.X
    
    # Normalise images to values in between 0 and 1
    def normalize_X(self) :
        self.X = self.store_Image_Array()
        X_min = self.X.min(axis=(0, 1), keepdims=True)
        X_max = self.X.max(axis=(0, 1), keepdims=True)
        self.X_norm = (self.X - X_min)/(X_max - X_min)
        return self.X_norm
    
    # Read actual image numbers
    def data_Read(self) :
        return pd.read_csv(self.csv, header = None)
    
    # One hot encode actual image numbers
    def one_Hot(self) :
        i = 0
        for a in self.data:
            if a == 10 :
                a = 0
            self.y[i, a] = 1
            i += 1
        return self.y
    
    # Convert actual image numbers to binary array
    def binary_One_Hot(self):
        
        i = 0
        for a in self.data :
            binary_str = '{0:20b}'.format(int(a))

            for b in range(0,self.binary_digits) :
                if binary_str[b] == ' ' :
                    self.y_bin[i, b] = 0
                else :
                    self.y_bin[i, b] = int(binary_str[b])   
            i += 1
        return self.y_bin
        
        
        
        
        
        
        