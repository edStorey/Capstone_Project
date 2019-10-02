from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras import optimizers
from keras.callbacks import History 
import numpy as np


class digitModel():

    def __init__(self,  X, y, X_test, y_test, data, height, width, classes, lr = 0.0003):
        self.classes = classes
        self. X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.height = height
        self.width = width
        self.data = data
        self.model = Sequential()
        self.history = History()
        self.learn_rate = lr
        self.model_Structure()
        self.model_Train()
        self.predictions = self.model_Pred()
        self.performance = self.model_Performance()
    
    # Define Neural Net structure
    def model_Structure(self):
        
        self.model.add(Conv1D(128, 8, activation = 'relu', input_shape = (self.height, self.width), 
                 kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform'))
        self.model.add(Conv1D(128, 8, activation = 'relu', kernel_initializer = 'random_normal', 
                 bias_initializer = 'random_normal'))
        self.model.add(Conv1D(128, 8, activation = 'relu', kernel_initializer = 'random_normal', 
                 bias_initializer = 'random_normal'))
        self.model.add(Conv1D(128, 8, activation = 'relu', kernel_initializer = 'random_normal', 
                 bias_initializer = 'random_normal'))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation = 'relu', kernel_initializer = 'random_uniform', 
                bias_initializer = 'random_normal'))
                       
        self.model.add(Dense(self.classes))
        self.model.add(Activation("sigmoid"))
        
    # Train on intput data    
    def  model_Train(self):
        
        sgd = optimizers.SGD(lr= self.learn_rate)
        
        self.model.compile(loss="categorical_crossentropy", optimizer= sgd, metrics=["accuracy"])
        
        self.history = self.model.fit(x = self.X, y =self.y, epochs = 128, batch_size = 256, shuffle = True)

    # Predictions for test data    
    def model_Pred(self):
        return self.model.predict(self.X_test)
    
    
    # Test predictions output
    def model_Performance(self):
        
        self.predictions = self.model_Pred()
        pred_len = len(self.predictions)
        test_predictions = np.zeros(pred_len)
        
        for i in range(0, pred_len) :
            test_predictions[i] = 1 if np.argmax(self.predictions[i, :]) == self.data[i] else 0
            
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
        