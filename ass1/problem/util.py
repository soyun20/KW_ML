# -*- coding: utf-8 -*-

import numpy as np
      
def feature_normalization(data): # 10 points
    # parameter 
    feature_num = data.shape[1]
    data_point = data.shape[0]
    
    # you should get this parameter correctly
    normal_feature = np.zeros([data_point, feature_num])
    mu = np.zeros([feature_num])
    std = np.zeros([feature_num])
    
    # your code here

    # end

    return normal_feature
        
def split_data(data, label, split_factor):
    return  data[:split_factor], data[split_factor:], label[:split_factor], label[split_factor:]

def get_normal_parameter(data, label, label_num): # 20 points
    # parameter
    feature_num = data.shape[1]
    
    # you should get this parameter correctly    
    mu = np.zeros([label_num,feature_num])
    sigma = np.zeros([label_num,feature_num])

    # your code here

    # end
    
    return mu, sigma

def get_prior_probability(label, label_num): # 10 points
    # parameter
    data_point = label.shape[0]
    
    # you should get this parameter correctly
    prior = np.zeros([label_num])
    
    # your code here

    # end
    return prior

def Gaussian_PDF(x, mu, sigma): # 10 points
    # calculate a probability (PDF) using given parameters
    # you should get this parameter correctly
    pdf = 0
    
    # your code here

    # end
    
    return pdf

def Gaussian_Log_PDF(x, mu, sigma): # 10 points
    # calculate a probability (PDF) using given parameters
    # you should get this parameter correctly
    log_pdf = 0
    
    # your code here

    # end
    
    return log_pdf

def Gaussian_NB(mu, sigma, prior, data): # 40 points
    # parameter
    data_point = data.shape[0]
    label_num = mu.shape[0]
    
    # you should get this parameter correctly   
    likelihood = np.zeros([data_point, label_num])
    posterior = np.zeros([data_point, label_num])
    ## evidence can be ommitted because it is a constant
    
    # your code here
        ## Function Gaussian_PDF or Gaussian_Log_PDF should be used in this section
    # end
    return posterior

def classifier(posterior):
    data_point = posterior.shape[0]
    prediction = np.zeros([data_point])
    
    prediction = np.argmax(posterior, axis=1)
    
    return prediction
        
def accuracy(pred, gnd):
    data_point = len(gnd)
    
    hit_num = np.sum(pred == gnd)

    return (hit_num / data_point) * 100, hit_num

    ## total 100 point you can get 