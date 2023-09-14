# -*- coding: utf-8 -*-

import numpy as np


def feature_normalization(data):  # 10 points
    # parameter
    feature_num = data.shape[1]
    data_point = data.shape[0]

    # you should get this parameter correctly
    normal_feature = np.zeros([data_point, feature_num])
    mu = np.zeros([feature_num])
    std = np.zeros([feature_num])

    # your code here
    # calculate the mean and standard deviation
    mu = np.mean(data, 0) # axis = 0
    std = np.std(data, 0)
    # calculate normal_feature ((x - m)/sigma)
    normal_feature = (data - mu) / std
    # end

    return normal_feature


def split_data(data, label, split_factor): # split train data and test data
    return data[:split_factor], data[split_factor:], label[:split_factor], label[split_factor:]


def get_normal_parameter(data, label, label_num):  # 20 points
    # parameter
    feature_num = data.shape[1] # 4

    # you should get this parameter correctly
    mu = np.zeros([label_num, feature_num])
    sigma = np.zeros([label_num, feature_num])

    # your code here
    # Store data values of data by label in a tmp array
    tmp = np.zeros((label_num,), dtype=np.ndarray)
    for i in range(label_num):
        cond = np.where(label == i) # Find a location index that satisfies that label
        tmp[i] = np.zeros((cond[0].shape[0], feature_num)) # Create as many arrays as the number of data that have that label

    for tag in range(label_num):
        idx = 0
        cond = np.where(label == tag) # Find the data that fits the label
        if len(cond[0]) > 0: # Number of data indexes by label
            for i in cond[0]:
                tmp[tag][idx] = data[i] # Store data in a tmp array for each label
                idx += 1
            # Calculate the mean and standard deviation of data for each level
            mu[tag] = np.mean(tmp[tag], 0)
            sigma[tag] = np.std(tmp[tag], 0)
        else: # If no such label data exists
            # Save nan Value
            mu[tag] = np.nan
            sigma[tag] = np.nan
    # end

    return mu, sigma


def get_prior_probability(label, label_num):  # 10 points
    # parameter
    data_point = label.shape[0] # 100

    # you should get this parameter correctly
    prior = np.zeros([label_num])

    label = list(label)
    # your code here
    # Obtain the priority value
    # the number of data belonging to the label / the number of total data
    for i in range(label_num):
        prior[i] = list(label).count(i)/data_point
    # end
    return prior


def Gaussian_PDF(x, mu, sigma):  # 10 points
    # calculate a probability (PDF) using given parameters
    # you should get this parameter correctly
    pdf = 0

    # your code here
    # Gaussian distribution expression
    pdf = (1/np.sqrt(2*np.pi * sigma**2)) * np.exp(-(x-mu)**2 / (2*sigma**2))
    # end

    return pdf


def Gaussian_Log_PDF(x, mu, sigma):  # 10 points
    # calculate a probability (PDF) using given parameters
    # you should get this parameter correctly
    log_pdf = 0

    # your code here
    # Log Gaussian Distribution Expression using Gaussian_PDF
    log_pdf = np.log(Gaussian_PDF(x, mu, sigma))
    # end

    return log_pdf


def Gaussian_NB(mu, sigma, prior, data):  # 40 points
    # parameter
    data_point = data.shape[0] # 50 (test_data)
    label_num = mu.shape[0] # 3

    # you should get this parameter correctly
    likelihood = np.ones([data_point, label_num]) # matrix
    posterior = np.zeros([data_point, label_num]) # matrix
    ## evidence can be ommitted because it is a constant

    # your code here
    ## Function Gaussian_PDF or Gaussian_Log_PDF should be used in this section

    for i in range(data_point): # test data row
        for j in range(label_num): # class (label)
            for k in range(data.shape[1]): # feature_num
                # calculate the likelihood of data values according to each label
                likelihood[i][j] *= Gaussian_PDF(data[i][k], mu[j][k], sigma[j][k])
                if np.isnan(likelihood[i][j]): # If the data is nan, store the largest negative value
                    posterior[i][j] = -np.inf
                else: # posterior = log(likelihood * prior)
                    posterior[i][j] = np.log1p(likelihood[i][j]*prior[j])
    # end
    return posterior


def classifier(posterior):
    data_point = posterior.shape[0] # 50 (test)
    prediction = np.zeros([data_point])

    # classify by labels with large posterior values
    prediction = np.argmax(posterior, axis=1)

    return prediction


def accuracy(pred, gnd):
    data_point = len(gnd)

    # Calculate accuracy against predicted values and correct answers
    hit_num = np.sum(pred == gnd)

    return (hit_num / data_point) * 100, hit_num

    ## total 100 point you can get