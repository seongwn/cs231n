from builtins import range
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_dims = W.shape[0]
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)
        exp_scores = np.exp(scores-np.max(scores))
        prob_scores = exp_scores/np.sum(exp_scores)
        loss += -np.log(prob_scores[y[i]])
        
        for d in range(num_dims):
            for k in range(num_classes):
                if k == y[i]:
                    dW[d, k] += X.T[d, i] * (prob_scores[k]-1)
                else:
                    dW[d, k] += X.T[d, i] * prob_scores[k]
    
        

    loss /= num_train
    loss +=  reg * np.sum(W*W)
  
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #Get Loss
    scores = X.dot(W)
    scores[np.arange(scores.shape[0])] -= np.reshape(np.max(scores,axis=1),[-1,1])
    exp_scores = np.exp(scores)
    scores_sum = np.reshape(np.sum(exp_scores,axis=1),[-1,1])
    prob_scores = exp_scores / scores_sum
    loss_i = -np.log(prob_scores[np.arange(scores.shape[0]),y])
    loss = np.sum(loss_i)
    
    #Get DW
    correct_class = np.zeros((X.shape[0],dW.shape[1]))
    correct_class[np.arange(X.shape[0]),y] = -1
    dW += (X.T).dot(prob_scores)
    dW += (X.T).dot(correct_class)
    
    #regularization
    loss /= X.shape[0]
    loss += reg * np.sum(W*W)
    dW /= X.shape[0]
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
