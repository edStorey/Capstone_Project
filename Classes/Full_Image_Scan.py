from keras import models
import numpy as np


class image_Scan():
    
    def __init__(self, X, model_Path, model_Weights, width = 96, height = 64, size = 32, classes = 10, vert_Win_Extra = 2, 
                 hor_Win_Extra = 2) :
        self.X = X
        self.width = width
        self.height = height
        self.size = size
        self.classes = classes
        self.hor_Win_Extra = hor_Win_Extra
        self.vert_Win_Extra = vert_Win_Extra
        self.model_Path = model_Path
        self.model_Weights = model_Weights
        self.vertical_Windows = 0
        self.horizontal_Windows = 0
        self.y_dist = 32
        self.x_dist = 32
        self.array_size = 0
        self.predictions = self.caclulate_Vars()
        self.model = models.load_model(model_Path)
        self.load_Weights()
        self.scan_Model()
        
    # Calculate necessary variables    
    def caclulate_Vars(self) :
        horiz_scan_base = self.width/self.size
        vert_scan_base = self.height/self.size
        
        total_hor_win = self.hor_Win_Extra * horiz_scan_base
        total_vert_win = self.vert_Win_Extra * vert_scan_base
        
        self.y_dist = self.height/total_vert_win
        self.x_dist = self.width/total_hor_win
        
        offset = 1
        
        self.horizontal_Windows = total_hor_win - offset
        self.vertical_Windows = total_vert_win - offset
        total_windows = self.horizontal_Windows * self.vertical_Windows
        self.array_size = int(total_windows * self.classes)
        
        
        return np.zeros((len(self.X), self.array_size))
        
    # Load 32x32 digit detector
    def load_Weights(self) :
        self.model.load_weights(self.model_Weights)
        
        
    # Scan model through full images    
    def scan_Model(self) :
        cut_image = np.array(np.ones((1, self.size, self.size)))
        for i in range(0,len(self.X)) :
            count = 0
            for y in range(0, int(self.vertical_Windows)) : 
                y_1 = int(self.y_dist*y)
                y_2 = y_1 + self.size
                for x in range(0, int(self.horizontal_Windows)) :
                    # corner calculations
                    x_1 = int(self.x_dist * x) 
                    x_2 = x_1 + self.size
                    cut_image = np.array(self.X[i, y_1: y_2, x_1:x_2])
                    cut_image = np.resize(cut_image,[1, self.size, self.size])
                    pred_32 = self.model.predict(cut_image)

                    arrx_0 = int(self.classes*count)
                    arrx_1 = int(self.classes*(count+1))
                    self.predictions[i, arrx_0:arrx_1] = pred_32
                    count +=1