"""Utilities for the neural network modules
"""
# Author: Issam H. Laradji <issam.laradji@gmail.com>
# License: BSD 3 clause

import arrayfire as af
#import numpy as np
import cupy as np
import numpy
from af_type_utils import typemap

def logistic_sigmoid(x):
    #return 1 / (1 + af.exp(-x))
    return 1 / (1 + np.exp(-x))

def xlogy(x, y):
    return x * af.log(y)


def identity(X):
    """Simply return the input array.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Data, where n_samples is the number of samples
        and n_features is the number of features.
    Returns
    -------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Same as the input data.
    """
    return X


def logistic(X):
    """Compute the logistic function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    return logistic_sigmoid(X, out=X)


def tanh(X):
    """Compute the hyperbolic tan function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    return np.tanh(X, out=X)


def relu(X):
    """Compute the rectified linear unit function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    #np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    ii = (X < 0)
    if len(ii) > 0:
        X[ii] = 0
    return X


def softmax(X):
    """Compute the K-way softmax function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    """
    tmp = X - af.max(X, dim=1)
    X = af.exp(tmp)
    X /= af.sum(X, dim=1)

    return X


ACTIVATIONS = {'identity': identity, 'tanh': tanh, 'logistic': logistic,
               'relu': relu, 'softmax': softmax}


def inplace_identity_derivative(Z, delta):
    """Apply the derivative of the identity function: do nothing.
    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the identity activation function during
        the forward pass.
    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    # Nothing to do


def inplace_logistic_derivative(Z, delta):
    """Apply the derivative of the logistic sigmoid function.
    It exploits the fact that the derivative is a simple function of the output
    value from logistic function.
    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the logistic activation function during
        the forward pass.
    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta *= Z
    delta *= (1 - Z)


def inplace_tanh_derivative(Z, delta):
    """Apply the derivative of the hyperbolic tanh function.
    It exploits the fact that the derivative is a simple function of the output
    value from hyperbolic tangent.
    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the hyperbolic tangent activation
        function during the forward pass.
    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta *= (1 - Z ** 2)


def inplace_relu_derivative(Z, delta):
    """Apply the derivative of the relu function.
    It exploits the fact that the derivative is a simple function of the output
    value from rectified linear units activation function.
    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the rectified linear units activation
        function during the forward pass.
    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta[Z == 0] = 0


DERIVATIVES = {'identity': inplace_identity_derivative,
               'tanh': inplace_tanh_derivative,
               'logistic': inplace_logistic_derivative,
               'relu': inplace_relu_derivative}


def squared_loss(y_true, y_pred):
    """Compute the squared loss for regression.
    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) values.
    y_pred : array-like or label indicator matrix
        Predicted values, as returned by a regression estimator.
    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    return af.mean(af.flat((y_true - y_pred) ** 2)) / 2


def log_loss(y_true, y_prob):
    """Compute Logistic loss for classification.
    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.
    y_prob : array-like of float, shape = (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.
    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
#    eps = np.finfo(y_prob.dtype).eps
#    y_prob = np.clip(y_prob, eps, 1 - eps)
#    if y_prob.shape[1] == 1:
#        y_prob = np.append(1 - y_prob, y_prob, axis=1)
#
#    if y_true.shape[1] == 1:
#        y_true = np.append(1 - y_true, y_true, axis=1)
#
#    return - xlogy(y_true, y_prob).sum() / y_prob.shape[0]

    eps = numpy.finfo(typemap(y_prob.dtype())).eps
    y_prob[y_prob < eps] = eps
    y_prob[y_prob > (1.0 - eps)] = 1.0 - eps

    if y_prob.numdims() == 1:
        y_prob = af.join(1, (1.0 - y_prob).as_type(y_prob.dtype()), y_prob)

    if y_true.numdims() == 1:
        y_true = af.join(1, (1.0 - y_true).as_type(y_true.dtype()), y_true)

    return - af.sum(af.flat(xlogy(y_true, y_prob))) / y_prob.shape[0]



def binary_log_loss(y_true, y_prob):
    """Compute binary logistic loss for classification.
    This is identical to log_loss in binary classification case,
    but is kept for its use in multilabel case.
    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.
    y_prob : array-like of float, shape = (n_samples, 1)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.
    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    eps = np.finfo(y_prob.dtype).eps
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -(xlogy(y_true, y_prob) +
             xlogy(1 - y_true, 1 - y_prob)).sum() / y_prob.shape[0]


LOSS_FUNCTIONS = {'squared_loss': squared_loss, 'log_loss': log_loss,
                  'binary_log_loss': binary_log_loss}
