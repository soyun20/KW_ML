# -*- coding: utf-8 -*-

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans
# Libraries added without justification are a minus factor.

# Seed setting
seed_num = 2022
# np.random.seed(seed_num)
iteration = 100  # Number of times to repeat steps E and M.


class EM:
    """ expectation-maximization algorithm, EM algorithm
    The EM class is a class that implements an EM algorithm using GMM and kmeans.

    Within the fit function, the remaining functions should be used.
    Other functions can be added, but all tasks must be implemented with Python built-in functions and Numpy functions.
    You should annotate each function with a description of the function and parameters(Leave a comment).
    """

    def __init__(self, n_clusters, iteration):
        """
        Parameters
        ----------
        n_clusters (int): Num of clusters (num of GMM)
        iteration (int): Num of iteration
            Termination conditions if the model does not converge
        mean (ndarray): Num of clusters x Num of features
            The mean vector that each cluster has.
        sigma (ndarray): Num of clusters x Num of features x Num of features
            The covariance matrix that each cluster has.

        """
        self.n_clusters = n_clusters
        self.iteration = iteration
        self.mean = np.zeros((n_clusters, 4))
        self.sigma = np.zeros((n_clusters, 4, 4))
        self.pi = np.zeros((n_clusters))

    def initialization(self, data):
       """ 1.initialization, 10 points
       Initial values for mean, sigma, and pi should be assigned.
       It have a significant impact on performance.

       your comment here

       Initialize mean, sigma, pi
       Use np.random.choice to extract sample indexes and set a random mean value
       Sets the initial value of the sigma as a identity matrix.
       """
       # your code here

       # Initialize means to random data points
       indices = np.random.choice(data.shape[0], size=self.n_clusters, replace=False)
       self.mean = data[indices]

       # Initialize covariances to identity matrices
       self.sigma = np.stack([np.eye(data.shape[1]) for i in range(self.n_clusters)])

       # Initialize mixing coefficients to be uniform
       self.pi = np.ones(self.n_clusters) / self.n_clusters
       
       # return nothing (return something or nothing)

    def multivariate_gaussian_distribution(self, data, mean, cov):
       """ 2.multivariate_gaussian_distribution, 10 points
       Use the linear algebraic functions of Numpy. π of this function is not self.pi

       your comment here

       Implement the multivariate_gaussian_distribution formula
       Use np.linalg.det function to calculate determinants and use np.linalg.inv function to calculate inverse matrix

       """
       # your code here
       n_features = data.shape[1] # 4

       # Compute the determinant and inverse of the covariance matrix.
       det_cov = np.linalg.det(cov)
       inv_cov = np.linalg.inv(cov)

       # Compute the probability density values using the multivariate Gaussian formula.
       output = np.zeros(data.shape[0])
       for i in range(data.shape[0]):
           # Compute the difference between the data point and the mean of the cluster to which it belongs.
           x_minus_mean = data[i] - mean
           # Compute the exponent of the multivariate Gaussian distribution formula using the inverse of the covariance matrix.
           exponent = -0.5 * x_minus_mean.T @ inv_cov @ x_minus_mean
           # Compute the normalization constant of the multivariate Gaussian distribution formula.
           norm_const = np.sqrt((2*np.pi)**n_features * det_cov)
           # Compute the probability density value for the current data point and store it in the output array.
           output[i] = np.exp(exponent) / norm_const

       return output # something or nothing

    def expectation(self, data):
       """ 3.expectation step, 20 points
       The multivariate_gaussian_distribution(MVN) function must be used.

       your comment here

       gamma is posterior
       Using multivariate_gaussian_distribution, implement the formula posterior.
       """
       # your code here

       # Initalize gamma value
       gamma = np.zeros((data.shape[0], self.n_clusters))
       
       # Implement the formula posterior
       for k in range(self.n_clusters):
          gamma[:,k] = self.pi[k] *  self.multivariate_gaussian_distribution(data, self.mean[k], self.sigma[k])

       gamma = gamma / np.sum(gamma, axis = 1, keepdims=True)

       return gamma # something or nothing


    def maximization(self, data, gamma):
       """ 4.maximization step, 20 points
       Hint. np.outer

       your comment here

       Update pi, mean, sigma values
       When we calculate mean, reshape(-1, 1) is intended to dimension the data multiplied by gamma[:,k].
       """
       # your code here

       # Update pi
       self.pi = gamma.sum(axis=0) / len(data)

       # Update mean
       for k in range(self.n_clusters):
         self.mean[k] = np.sum(gamma[:,k].reshape(-1, 1) * data, axis = 0) / gamma[:,k].sum()

       n_features = data.shape[1]
       # Update sigma
       for k in range(self.n_clusters):
           sigma_sum = np.zeros((n_features, n_features))
           # iterate over each data point
           for i in range(data.shape[0]):
               # calculate the difference between the data point and the cluster mean
               data_diff = data[i] - self.mean[k]
               # calculate the outer product of the data difference with its transpose and add it to the sum
               sigma_sum += gamma[i, k] * np.outer(data_diff.T, data_diff)
           # divide the sum by the sum of the corresponding gamma values to get the new covariance matrix for the cluster
           self.sigma[k] = sigma_sum / np.sum(gamma[:, k])

       # return nothing (return something or nothing)

    def fit(self, data):
       """ 5.fit clustering, 20 points
       Functions initialization, expectation, and maximization should be used by default.
       Termination Condition. Iteration is finished or posterior is the same as before. (Beware of shallow copy)
       Prediction for return should be formatted. Refer to iris['target'] format.

       your comment here

       This method fits a Gaussian Mixture Model to the input data using the Expectation-Maximization (EM) algorithm.
       The algorithm runs for a specified number of iterations or until the posterior probabilities no longer change significantly.
       The method returns predicted cluster labels for the input data based on the final model fit.
       """
       # your code here
       # initialization
       self.initialization(data)

       # initialize likelihood
       prev_likelihood = -np.inf

       for i in range(self.iteration):
           # E step
           gamma = self.expectation(data)

           # M step
           self.maximization(data, gamma)

           # compute log likelihood
           likelihood = np.log(np.sum([self.pi[k] * self.multivariate_gaussian_distribution(data, self.mean[k], self.sigma[k]) for k in range(self.n_clusters)]))
            
           # check convergence
           if np.abs(likelihood - prev_likelihood) < 1e-6:
               break
                
           prev_likelihood = likelihood
            
       # return predicted labels
       labels = np.argmax(gamma, axis=1)
       return labels

def plotting(data, title):
  """ 6.plotting, 20 points with report
   Default = seaborn pairplot

   your comment here

   Generate Graphs
   """
   # your code here

  # Specify a fixed color for each cluster.
  if(title=="Origin"):
    palette = {'setosa': "blue", 'versicolor': "orange", 'virginica': "green"}
  else:
    palette = {0: "blue", 1: "orange", 2: "green"}

  # Creates a figure with a 4x4 grid of subplots and assigns it to the variables fig and axes. It also sets the figure title to the title argument.
  fig, axes = plt.subplots(4, 4, figsize=(10, 10))
  fig.suptitle(title, fontsize = 16)

  # This code loops through each pair of columns in the data argument (excluding the last column)
  # Generates either a KDE plot or a scatter plot in each corresponding subplot, depending on whether the columns are the same or different.
  for i, col1 in enumerate(data.columns[:-1]):
    for j, col2 in enumerate(data.columns[:-1]):
      ax = axes[i][j]
      if i == j:
        sns.kdeplot(data=data, x=col2, hue="labels", fill=True, ax=ax, palette=palette)
      else:
        sns.scatterplot(data=data, x=col2, y=col1, hue="labels", ax=ax, palette=palette)

  plt.tight_layout()

  # return nothing (return something or nothing)




if __name__ == '__main__':
    # Loading and labeling data
    iris = datasets.load_iris()
    original_data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['labels'])
    original_data['labels'] = original_data['labels'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    plotting(original_data, "Origin")

    # Only data is used W/O labels beacause EM and Kmeans are unsupervised learning
    data=iris['data']

    # Unsupervised learning(clustering) using EM algorithm
    EM_model = EM(n_clusters=3, iteration=iteration)
    EM_pred = EM_model.fit(data)
    EM_pd=pd.DataFrame(data=np.c_[data, EM_pred], columns=iris['feature_names'] + ['labels'])
    plotting(EM_pd, "EM")

    # Why are these two elements almost the same? Write down the reason in your report. Additional 10 points
    print(f'pi :            {EM_model.pi}')
    print(f'count / total : {np.bincount(EM_pred) / 150}')

    # Unsupervised learning(clustering) using KMeans algorithm
    KM_model = KMeans(n_clusters=3, init='random', random_state=seed_num, max_iter=iteration).fit(data)
    KM_pred = KM_model.predict(data)
    KM_pd = pd.DataFrame(data=np.c_[data, KM_pred], columns=iris['feature_names'] + ['labels'])
    plotting(KM_pd, "KMeans")

    # No need to explain.
    for idx in range(2):
        EM_point = np.argmax(np.bincount(EM_pred[idx * 50:(idx + 1) * 50]))
        KM_point = np.argmax(np.bincount(KM_pred[idx * 50:(idx + 1) * 50]))
        EM_pred = np.where(EM_pred == idx, 3, EM_pred)
        EM_pred = np.where(EM_pred == EM_point, idx, EM_pred)
        EM_pred = np.where(EM_pred == 3, EM_point, EM_pred)
        KM_pred = np.where(KM_pred == idx, 3, KM_pred)
        KM_pred = np.where(KM_pred == KM_point, idx, KM_pred)
        KM_pred = np.where(KM_pred == 3, KM_point, KM_pred)

    EM_hit = np.sum(iris['target'] == EM_pred)
    KM_hit = np.sum(iris['target'] == KM_pred)
    print(f'EM Accuracy: {round(EM_hit / 150, 2)}    Hit: {EM_hit} / 150')
    print(f'KM Accuracy: {round(KM_hit / 150, 2)}    Hit: {KM_hit} / 150')
