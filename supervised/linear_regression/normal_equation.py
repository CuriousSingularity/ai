# linear regression with normal equation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from itertools import combinations


class DataPreprocessor():
    """
    Data preprocessor class, which supports:
    - loading data
    - feature scaling 
    - normalize
    - adding bias term  
    """
    
    def __init__(self):
        pass
    
    def normalize(self, data : np.array) -> np.array:
        """
        2D normalization:
            Data = (Data - Mean)/standard_deviation
        
        Args:
            data : 2d numpy array
            
        Return
            normalized 2d numpy array data
        """

        mean = []
        standard_deviation = []

        sample_size, feature_size = data.shape

        for i in range(feature_size):
            each_mean = np.mean(data[:,i])
            each_sd = np.sd(data[:,i]) # can also use : each_sd = np.max(data[:,i]) - np.min(data[:,i])
            mean.append(each_mean)
            standard_deviation.append(each_sd)

        normalized_data = (data - mean) / standard_deviation

        return normalized_data
    
    
    def add_bias_term(self, X : np.array) -> np.array:
        """
        Add Bias Term
        
        Args:
            data : 2d numpy array
            
        Return
            extra bias column + data
        """
        bias = np.ones((X.shape[0],1))
        X = np.append(bias, X, axis=1)
        
        return X
    
    
    def feature_scaling(self, X : np.array) -> np.array:
        """
        Feature scaling
        Data = (Data - Mean)/standard_deviation
        
        Args:
            data : 2d numpy array
            
        Return
            normalized 2d numpy array data
        """
        X_norm = self.normalize(X)
        
        return X_norm
    

    def data_load(self, file : str, sep : str = ';') -> (tuple, tuple):
        """
        Load data from file
        
        Args:
            data : 2d numpy array
            
        Return
            data : (samples, labels)
        """
        df = pd.read_csv(file, sep)

        df_X = df[["dim_x","dim_y","ch","kernel_x","kernel_y"]]
        df_Y = df[["CPU Cycles"]]

        X = df_X.to_numpy()
        Y = df_Y.to_numpy()

        return X, Y
    
    
    def generate_polynominal_features(self, X : np.array) -> np.array:
        """
        Generate complex polynominal features from Data from different combinations
        
        Args:
            data : 2d numpy array
            
        Return
            complex data features 
        """
        sample_size, feature_size = X.shape
        
        gen_possible_indices = []
        
        for i in range(feature_size+1):
            for subset in combinations(range(feature_size), i):
                if len(subset):
                    gen_possible_indices.append(subset)
        
        new_features = []
        for each_combination in gen_possible_indices:
            
            new_feature = np.ones((sample_size))
            # multiple features to generate addition features
            for index in each_combination:
                new_feature *= X[:,index]
                
            new_features.append(new_feature) 
        
        X_new = np.array(new_features).T
        
        return X_new
        

class LR_NormalEquation(DataPreprocessor):
    """
    Computes the Linear Regression with Normal Equation from Log-Maximum Likehood estimation.
    
    parameter or theta = ((X^T.X)^-1).(X^T.Y)
    
    TODO: Add base "class model" which has the basic functionalities : 
        - train,
        - validate, 
        - predict, 
        - error calculation
    """
    def train(self, X, Y):
        """
        Trains the model for linear regression with normal equation.
        
        Args:
            X : input samples
            Y : labels
            
        Return:
            trained parameter or theta
        """
        
        first_term = np.linalg.inv(np.dot(X.T, X))
        second_term = np.dot(X.T, Y)
        theta =  np.dot(first_term, second_term)
        
        return theta
    
    
    def predict(self, X_test, theta):
        """
        Predict the values from model theta for unseen data X.
        
        Args:
            X : samples
            theta : parameter of trained model
            
        Return:
            predicted value for test data
        """
        
        Y_test = np.dot(X_test, theta)
        
        return Y_test
    
    def error(self, Y, Y_test):
        """
        Calculate the error between the actual value and predicted values.
        
        Args:
            Y : labels
            Y_test : predicted values
            
        Return:
            relative error, average relative error, maximum relative error
        """
        relative_error = 1 - Y/Y_test
        avg_relative_error = np.sum(relative_error) / len(Y)
        max_relative_error = np.max(relative_error)
        
        return relative_error, avg_relative_error, max_relative_error


def Main():
    # read dataset file
    filename = "dataset.csv"

    lr_model = LR_NormalEquation()

    # not splitting into train and test sets as the data is very less
    X, Y = lr_model.data_load(filename)
    print(f"Original Data X = {X.shape}, Y = {Y.shape}")

    # create features
    X = lr_model.generate_polynominal_features(X)
    print(f"Generated Features Data X = {X.shape}, Y = {Y.shape}")

    #X = lr_model.feature_scaling(X)
    X = lr_model.add_bias_term(X)
    print(f"With Bias X = {X.shape}, Y = {Y.shape}")

    #train the model
    theta = lr_model.train(X, Y)

    #predict from the model
    #print(theta)

    Y_predicted = lr_model.predict(X, theta)

    fig, ax = plt.subplots()
    ax.plot(Y, color='g', label='label')
    ax.plot(Y_predicted, color='b', label='prediction')
    plt.show()

if __name__ == '__main__':
    Main()





