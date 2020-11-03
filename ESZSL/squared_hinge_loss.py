import numpy as np

def squared_hinge_loss(AL, Y):
    """
    Implement the squared hinge loss defined by max(0, 1-y.^y)² 

    Arguments:
    AL -- probability vector corresponding to label predictions, shape (1, number of examples)
    Y -- true "label" vector, shape (1, number of examples)

    Returns:
    cost -- squared hinge loss
    """

    # Compute loss from aL and y
    cost = np.sum(np.maximum(0, 1-Y*AL)**2)

    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost

AL = np.array([0.98,0.99,0.43,0.21])
Y = np.array([1,1,1,1])

print("Cost = " + str(squared_hinge_loss(AL, Y)))