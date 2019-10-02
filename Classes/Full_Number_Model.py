from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras import optimizers
import numpy as np

class fullNumberModel():

    def __init__(self, X, y, X_test, binary_Data, length, classes, lr = 0.0003):
        self.classes = classes
        self. X = X
        self.y = y
        self.X_test = X_test
        self.length = length
        self.data = binary_Data
        self.classes = classes
        self.model = Sequential()
        self.learn_rate = lr
        self.model_Structure()
        self.model_Train()
        self.predictions = self.model_Pred()
        self.performance = self.model_Performance()
    
    # Define Neural Net structure
    def model_Structure(self):
        
        self.model.add(Dense(128, activation = 'relu', input_shape = (self.length,)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation = 'relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation = 'relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(512, activation = 'relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation = 'relu'))
        self.model.add(Dropout(0.5))
                       
        self.model.add(Dense(self.classes))
        self.model.add(Activation('sigmoid'))
        
     # Train on intput data    
    def  model_Train(self):
        
        sgd = optimizers.SGD(lr= self.learn_rate)
        
        self.model.compile(loss="categorical_crossentropy", optimizer= sgd, metrics=["accuracy"])
        
        self.model.fit(x = self.X, y =self.y, epochs = 1000, batch_size = 1024, shuffle = True)
        
    # Predictions for test data    
    def model_Pred(self):
        return self.model.predict(self.X_test)
    
    # Predictions for test data 
    def model_Performance(self):
        pred_len = len(self.predictions)
        test_predictions = np.ones(pred_len)
        
        for i in range(0, pred_len) :
            for a in range(0, len(self.data[i, :])):
                if ((np.round(self.predictions[i, a]) == self.data[i, a]) and test_predictions[i] == 1) : 
                    test_predictions[i] = 1
                else :
                    test_predictions[i] = 0
                    break

        return test_predictions
        
    # Save model structure 
    def model_Save(self):
        return self.model.save('Model_Data/model_32x32.h5') 
    
    # Save Weights
    def model_Save_Weights(self):
        return self.model.save_weights('Model_Data/model_weights_32x32.h5')
    
    # Load previous model
    def model_Load(self):
        self.model = models.load_model('ModelData/model_32x32.h5') 
    
    # load previosu weights
    def model_Save_Weights(self):
        self.model.load_weights('ModelData/model_weights_32x32.h5') 
        