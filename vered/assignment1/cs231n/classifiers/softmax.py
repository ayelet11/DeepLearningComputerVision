import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.

  loss = 0.0
  dW = np.zeros_like(W)
  num_train = len(X)
  num_class = W.shape[1]
  L = np.zeros(num_train)

  scores = X.dot(W)

  for i in range(num_train):
    norm_scores = scores[i] - np.max(scores[i])
    L[i] = -np.log(np.exp(norm_scores[y[i]]) / np.sum(np.exp(norm_scores)))

    dW[:, y[i]] -= X[i]
    for j in range(num_class):
      dW[:, j] += X[i]*(np.exp(norm_scores[j]) / np.sum(np.exp(norm_scores)))

  loss = np.sum(L)/num_train + reg * np.sum(W * W)
  dW = dW/num_train + 2 * reg * W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  num_train = len(X)
  dW = np.zeros_like(W)

  scores = X.dot(W)

  scores = np.exp(scores - np.max(scores))
  loss = np.sum(-np.log(scores[range(X.shape[0]), y] / np.sum(scores, axis=1))) / num_train + reg * np.sum(W * W)

  score_ratio = scores/np.sum(scores, axis=1, keepdims=True)

  Y = np.zeros(( X.shape[0], W.shape[1]))
  Y[range(len(y)), y] = -1

  dW = np.transpose(X).dot(Y)
  dW += np.transpose(X).dot(score_ratio)

  dW = dW/num_train + 2 * reg * W


  return loss, dW

