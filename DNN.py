import numpy as np
import time
import pickle

def mlp_check_dimensions(x, y, ws, bs):
    """
    Return True if the dimensions in double_u and beta agree.
    :param x: a list of lists representing the x matrix.
    :param y: a list output values.
    :param ws: a list of weight matrices (one for each layer)
    :param bs: a list of biases (one for each layer)
    :return: True if the dimensions of x, y, ws and bs match
    """
    x = np.array(x)
    y = np.array(y)
    if x.shape[0] == y.shape[0]:
        if len(ws) == len(bs):
            for i, w in enumerate(ws):
            # check weight and bias vectors
                if ws[i].shape[1] == ws[i+1].shape[0]:
                    if ws[i].shape[1] == bs[i].shape[1]:
                        if ws[i+1].shape[1] == bs[i+1].shape[1]:
                            if x.shape[1] == ws[i].shape[0]:
                                return True
    if ws[-1].shape[1] == bs[-1].shape[1] == y.shape[1]:
        return True
    return False
    
## Forward Propagation Subroutines

def mlp_ReLU(z):
    """
    Return the Rectified Linear Unit
    :param z: the input "affinities" for this neuron.
    :return: hyperbolic tangent "squashing function"
    """
    return np.maximum(np.zeros(np.shape(z)), z)

def mlp_tanh(z):
    """
    Return the hyperbolic tangent of z using numpy.
    :param z: the input "affinities" for this neuron.
    :return: hyperbolic tangent "squashing function"
    """
    return np.tanh(z)

def mlp_softmax(z):
    """
    Return the softmax function at z using a numerically stable approach.
    :param z: A real number, list of real numbers, or list of lists of numbers.
    :return: The output of the softmax function for z with the same shape as z.
    """
    for i, z_i in enumerate(z):
        row_max = max(z_i)
        for j, z_j in enumerate(z_i):
            z[i][j] = z[i][j] - row_max
    h = z.copy()
    for i, z_i in enumerate(z):
        row_sum = 0
        for j, z_j in enumerate(z_i):
            row_sum += np.exp(z[i][j])
        for j, z_j in enumerate(z_i):
            h[i][j] = np.exp(z[i][j]) / row_sum
    return h

def mlp_linear_input(h, w, b):
    """
    Return the layer input z as a function of h, w, and b.
    :param h: the input from the previous layer.
    :param w: the weights for this layer.
    :param b: the biases for this layer.
    :return: the linear network activations for this layer.
    """
    z = h.dot(w) + b
    return z

def mlp_feed_layer(h, w, b, phi):
    """
    Return the output of a layer of the network.
    :param h: The input to the layer (output of last layer).
    :param w: The weight matrix for this layer.
    :param b: The bias vector for this layer.
    :param phi: The activation function for this layer.
    :return: The output of this layer.
    """
    h = phi(mlp_linear_input(h, w, b))
    return h

def mlp_feed_forward(x, ws, bs, phis):
    """
    Return the output of each layer of the network.
    :param x: The input matrix to the network.
    :param ws: The list of weight matrices for layers 1 to l.
    :param bs: The list of bias vectors for layers 1 to l.
    :param phis: The list of activation functions for layers 1 to l.
    :return: The list of outputs for layers 0 to l
    """
    hs = []
    hs.append(x)
    for i, phi_i in enumerate(phis):
        hs.append(mlp_feed_layer(hs[i], ws[i], bs[i], phis[i]))
    return hs

## Prediction Subroutines

def mlp_predict_probs(x, ws, bs, phis):
    """
    Return the output matrix of probabilities for input matrix 'x'.
    :param x: The input matrix to the network.
    :param ws: The list of weight matrices for layers 1 to l.
    :param bs: The list of bias vectors for layers 1 to l.
    :param phis: The list of activation functions for layers 1 to l.
    :return: The output matrix of probabilities (p)
    """
    hs = mlp_feed_forward(x, ws, bs, phis)
    return hs[-1]
   
def mlp_predict(x, ws, bs, phis):
    """
    Return the output vector of labels for input matrix 'x'.
    :param x: The input matrix to the network.
    :param ws: The list of weight matrices for layers 1 to l.
    :param bs: The list of bias vectors for layers 1 to l.
    :param phis: The list of activation functions for layers 1 to l.
    :return: The output vector of class labels.
    """
    p = mlp_predict_probs(x, ws, bs, phis)
    y_hat = np.argmax(p, axis = 1)
    return y_hat

## Training Subroutines

def mlp_cost(x, y, ws, bs, phis, alpha):
  """
  Return the cross entropy cost function with L2 regularization term.
  :param x: a list of lists representing the x matrix.
  :param y: a list of lists of output values.
  :param ws: a list of weight matrices (one for each layer)
  :param bs: a list of biases (one for each layer)
  :param phis: a list of activation functions
  :param alpha: the hyperparameter controlling regularization
  :return: The cost function
  """

  #Cross Entropy Cost
  p = np.multiply(y, np.log(mlp_predict_probs(x, ws, bs, phis) + 1e-8))

  crossEntCost = - np.sum(p) / len(y)

  # Regularization Term
  regTerm = 0
  for i in range(len(ws)):
    for j in range(len(ws[i])):
      for k in range(len(ws[i][j])):
        regTerm += ws[i][j][k] ** 2
  regTerm *= alpha / 2

  return crossEntCost + regTerm

def mlp_propagate_error(x, y, ws, bs, phis, hs):
  """
  Return a list containing the gradient of the cost with respect to z^(k)
  for each layer.
  :param x: a list of lists representing the x matrix.
  :param y: a list of lists of output values.
  :param ws: a list of weight matrices (one for each layer)
  :param bs: a list of biases (one for each layer)
  :param phis: a list of activation functions
  :param hs: a list of outputs for each layer include h^(0) = x
  :return: A list of gradients of C with respect to z^(k) for k=1..l
  """
  bigD = []
  for i in range(len(hs) - 1, 0, -1):
    if (i == len(hs) - 1):
      bigD.append((hs[-1] - y) / len(y))
    else:
      dKPlusOne = bigD[len(hs) - (i + 2)]
      if (phis[i] == mlp_tanh):
        #tanh
        hSqrd = hs[i] ** 2
  
        matMUL = np.matmul(dKPlusOne, np.transpose(ws[i]))
  
        compMUL = np.multiply(matMUL, 1 - hSqrd)
      else:
        #ReLU
        hSqrd = hs[i]
        hSqrd[hSqrd > 0] = 1
  
        matMUL = np.matmul(dKPlusOne, np.transpose(ws[i]))

        compMUL = np.multiply(matMUL, hSqrd)
      
      bigD.append(compMUL)
  returnArray = []
  for i in range(len(bigD) - 1, -1, -1):
      returnArray.append(bigD[i])
  return returnArray

def mlp_gradient(x, y, ws, bs, phis, alpha):
  """
  Return a list containing the gradient of the cost with respect to z^(k)
  for each layer.
  :param x: a list of lists representing the x matrix.
  :param y: a list of lists of output values.
  :param ws: a list of weight matrices (one for each layer)
  :param bs: a list of biases (one for each layer)
  :param phis: a list of activation functions
  :param alpha: the regularization term
  :return: A list of gradients of J with respect to z^(k) for k=1..l
  """
  hss = mlp_feed_forward(x, ws, bs, phis)
  ds = mlp_propagate_error(x, y, ws, bs, phis, hss)
  gws = []
  gbs = []
  for k in range(0, len(hss) - 1):
    gws.append(np.matmul(np.transpose(hss[k]), ds[k]) + alpha * ws[k])
    gbs.append(np.matmul(np.transpose(np.ones(len(x))), ds[k]))
  return [gws, gbs]

## Learning Rate Scheduling

def mlp_constant_eta(iteration, eta_details):
  """
  Provides a constant learning rate.
  :param iteration: the current iteration of gradient decent
  :param eta_details[0]: the desired constant learning rate
  :return: a constant learning rate equal to the input
  """
  return eta_details[0]

def mlp_step_based_eta(iteration, eta_details):
  """
  Decreases the learning rate by proportion d every r iterations
  of the training method.
  :param iteration: the current iteration of gradient decent
  :param eta_details[0]: the initial learning rate (\eta_0)
  :param eta_details[1]: the droprate (r)
  :param eta_details[2]: the decay proportion (d)
  :return: the step-based learning rate for an iteration
  """
  return eta_details[0] * np.power(eta_details[2], np.floor((1 + iteration) / eta_details[1]))

def mlp_exponential_eta(iteration, eta_details):
  """
  Decreases the learning rate by proportion d every r iterations
  of the training method.
  :param iteration: the current iteration of gradient decent
  :param eta_details[0]: the initial learning rate (\eta_0)
  :param eta_details[1]: the decay rate (d)
  :return: the step-based learning rate for an iteration
  """
  return eta_details[0] * np.exp(-eta_details[1] * iteration)

## GD and SGD

def mlp_gradient_descent(x, y, ws0, bs0, phis, alpha, n_iter, eta_func, eta_details):
  """
  Uses gradient descent to estimate the weights, ws, and biases, bs,
  that reduce the cost.
  :param x: a list of lists representing the x matrix.
  :param y: a list of lists of output values.
  :param ws0: a list of initial weight matrices (one for each layer)
  :param bs0: a list of initial biases (one for each layer)
  :param phis: a list of activation functions
  :param alpha: the hyperparameter controlling regularization
  :param n_iter: the number of iterations
  :param eta_func: the learning rate schedule
  :param eta_details: hyperparameters for the learning rate schedule
  :return: the estimate weights, the estimated biases, record of cost
  """
  W = ws0.copy()
  B = bs0.copy()
  C = np.zeros(n_iter)
  
  for i in range(n_iter):
    # Record cost for analysis
    C[i] = mlp_cost(x, y, W, B, phis, alpha)
    
    gws, gbs = mlp_gradient(x, y, W, B, phis, alpha)
    for k in range(len(W)):
      W[k] -= eta_func(k, eta_details) * gws[k]
      B[k] -= eta_func(k, eta_details) * gbs[k]
  return [W, B, C]

def mlp_gradient_descent_results(x, y, ws0, bs0, phis, alpha, n_iter, eta_func, eta_details):
  """
  Uses gradient descent to estimate the weights, ws, and biases, bs,
  that reduce the cost.
  :param x: a list of lists representing the x matrix.
  :param y: a list of lists of output values.
  :param ws0: a list of initial weight matrices (one for each layer)
  :param bs0: a list of initial biases (one for each layer)
  :param phis: a list of activation functions
  :param alpha: the hyperparameter controlling regularization
  :param n_iter: the number of iterations
  :param eta_func: the learning rate schedule
  :param eta_details: hyperparameters for the learning rate schedule
  :return: the estimate weights, the estimated biases, record of cost
  """
  W = ws0.copy()
  B = bs0.copy()
  C = np.zeros(n_iter)
  
  for i in range(n_iter):
    # Record cost for analysis
    C[i] = mlp_cost(x, y, W, B, phis, alpha)
    
    gws, gbs = mlp_gradient(x, y, W, B, phis, alpha)
    for k in range(len(W)):
      W[k] -= eta_func(k, eta_details) * gws[k]
      B[k] -= eta_func(k, eta_details) * gbs[k]
  return [W, B, C]
    
def mlp_SGD(x, y, t, ws0, bs0, phis, alpha, n_iter, eta_func, eta_details):
  """
  Uses stochastic gradient descent to estimate the weights, ws, and biases, bs,
  that reduce the cost.
  :param x: a list of lists representing the x matrix.
  :param y: a list of lists of output values.
  :param t: sample size of the minibatches.
  :param ws0: a list of initial weight matrices (one for each layer)
  :param bs0: a list of initial biases (one for each layer)
  :param phis: a list of activation functions
  :param alpha: the hyperparameter controlling regularization
  :param n_iter: the number of iterations
  :param eta_func: the learning rate schedule
  :param eta_details: hyperparameters for the learning rate schedule
  :return: the estimate weights, the estimated biases, record of cost
  """
  W = ws0.copy()
  B = bs0.copy()
  C = np.zeros(n_iter)
  
  # Set Seed for reproducibility
  np.random.seed(1)
  
  for i in range(n_iter):
    # Random shuffle
    perm = np.random.permutation(x.shape[0])

    # Record Cost for Analysis
    C[i] = mlp_cost(x, y, W, B, phis, alpha)

    # Minibatch sample
    x_minibatch = (x[perm])[0:t]
    y_minibatch = (y[perm])[0:t]
    
    gws, gbs = mlp_gradient(x_minibatch, y_minibatch, W, B, phis, alpha)
    for k in range(len(W)):
      W[k] -= eta_func(k, eta_details) * gws[k]
      B[k] -= eta_func(k, eta_details) * gbs[k]
  return [W, B, C]

def mlp_SGD_results(x, y_train, x_test, y_test, y, t, ws0, bs0, phis, alpha, n_iter, eta_func, eta_details):
  """
  Uses stochastic gradient descent to estimate the weights, ws, and biases, bs,
  that reduce the cost.
  :param x: a list of lists representing the x matrix.
  :param y: a list of lists of output values.
  :param t: sample size of the minibatches.
  :param ws0: a list of initial weight matrices (one for each layer)
  :param bs0: a list of initial biases (one for each layer)
  :param phis: a list of activation functions
  :param alpha: the hyperparameter controlling regularization
  :param n_iter: the number of iterations
  :param eta_func: the learning rate schedule
  :param eta_details: hyperparameters for the learning rate schedule
  :return: the estimate weights, the estimated biases, record of cost
  """
  W = ws0.copy()
  B = bs0.copy()
  TrainingError = np.zeros(n_iter)
  TestingError = np.zeros(n_iter)
  
  # Set Seed for reproducibility
  np.random.seed(1)
  
  for i in range(n_iter):
    # Random shuffle
    perm = np.random.permutation(x.shape[0])

    # FOR RESULTS ONLY ####
    y_hat_train = mlp_predict(x, W, B, phis) 
    y_hat_test = mlp_predict(x_test, W, B, phis) 
    # Training Error
    TrainingError[i] = 1-(y_hat_train == y_train).mean() 
    # Testing Error
    TestingError[i] = 1-(y_hat_test == y_test).mean() 
    #########################
    
    # Minibatch sample
    x_minibatch = (x[perm])[0:t]
    y_minibatch = (y[perm])[0:t]
    
    gws, gbs = mlp_gradient(x_minibatch, y_minibatch, W, B, phis, alpha)
    for k in range(len(W)):
      W[k] -= eta_func(k, eta_details) * gws[k]
      B[k] -= eta_func(k, eta_details) * gbs[k]
  return [W, B, [TrainingError, TestingError]]

# Benchmark Data Preparation

## MNIST

def mlp_MNIST_data_prep():
    """
    Return the MNIST dataset.
    :return: The x matrix, y vector, and the wide matrix version of y
    """
    x_train = np.load( './MNIST/training_data.npy' )
    y_train = np.load( './MNIST/training_labels.npy' )
    
    # create the Y training wide matrix
    m = x_train.shape[0]
    c = (max(y_train) + 1) ## number of categories
    y_matrix_train = np.zeros((m,c))
    y_matrix_train[(range(m), y_train.astype(int))] = 1
    
    # extraxt X and Y for testing
    x_test = np.load( './MNIST/test_data.npy' )
    y_test = np.load( './MNIST/test_labels.npy' )
    
    # Create Y test wise matrix
    m = x_test.shape[0]
    c = (max(y_test) + 1) ## number of categories
    y_matrix_test = np.zeros((m,c))
    y_matrix_test[(range(m), y_test.astype(int))] = 1
    
    # Normalize training X
    x_mean = np.mean(x_train, axis=0)
    x_stdev = np.std(x_train, axis=0)
    for i, k in enumerate(x_stdev):
        if k == 0:
            x_stdev[i] = 1
    x_train = (x_train - x_mean) / x_stdev
    x_test = (x_test - x_mean) / x_stdev
    
    return x_train, x_test, y_train, y_test, y_matrix_train, y_matrix_test

## CIFAR-10

def unpickle(file):
    """
    Return an "unpickled" file.
    :param file: file to be "unpickled"
    :return: the unpickled file
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding ='latin1')
    return dict

def mlp_CIFAR_10_data_prep():
    """
    Return the prepared data using the provided MNIST 1000 dataset.
    :return: The x matrix, y vector, and the wide matrix version of y
    """

    # Unpickle the data
    batch_1 = unpickle("CIFAR10/data_batch_1")
    batch_2 = unpickle("CIFAR10/data_batch_2")
    batch_3 = unpickle("CIFAR10/data_batch_3")
    batch_4 = unpickle("CIFAR10/data_batch_4")
    batch_5 = unpickle("CIFAR10/data_batch_5")
    test_batch = unpickle("CIFAR10/test_batch")
    
    # Extract X for training
    x_train = np.array(batch_1["data"])
    x_train = np.append(x_train, batch_2["data"], axis=0)
    x_train = np.append(x_train, batch_3["data"], axis=0)
    x_train = np.append(x_train, batch_4["data"], axis=0)
    x_train = np.append(x_train, batch_5["data"], axis=0)
    
    # Extract y for training
    y_train = np.array(batch_1["labels"])
    y_train = np.append(y_train, batch_2["labels"])
    y_train = np.append(y_train, batch_3["labels"])
    y_train = np.append(y_train, batch_4["labels"])
    y_train = np.append(y_train, batch_5["labels"])
    
    # create the Y training wide matrix
    m = x_train.shape[0]
    c = (max(y_train) + 1) ## number of categories
    y_matrix_train = np.zeros((m,c))
    y_matrix_train[(range(m), y_train.astype(int))] = 1
    
    # extraxt X and Y for testing
    x_test = np.array(test_batch["data"])
    y_test = np.array(test_batch["labels"])
    
    # Create Y test wise matrix
    m = x_test.shape[0]
    c = (max(y_test) + 1) ## number of categories
    y_matrix_test = np.zeros((m,c))
    y_matrix_test[(range(m), y_test.astype(int))] = 1
    
    # Normalize training X
    x_mean = np.mean(x_train, axis=0)
    x_stdev = np.std(x_train, axis=0)
    for i, k in enumerate(x_stdev):
        if k == 0:
            x_stdev[i] = 1
    x_train = (x_train - x_mean) / x_stdev
    x_test = (x_test - x_mean) / x_stdev
    
    return x_train, x_test, y_train, y_test, y_matrix_train, y_matrix_test

## CIFAR-100

def mlp_CIFAR_100_data_prep():
    """
    Return the prepared data using the provided MNIST 1000 dataset.
    :return: The x matrix, y vector, and the wide matrix version of y
    """

    # Unpickle the data
    train_batch = unpickle("CIFAR100/train")
    test_batch = unpickle("CIFAR100/test")
    
    # Extract X for training
    x_train = np.array(train_batch["data"])
    
    # Extract y for training
    y_train = np.array(train_batch["fine_labels"])
    
    # create the Y training wide matrix
    m = x_train.shape[0]
    c = (max(y_train) + 1) ## number of categories
    y_matrix_train = np.zeros((m,c))
    y_matrix_train[(range(m), y_train.astype(int))] = 1
    
    # extraxt X and Y for testing
    x_test = np.array(test_batch["data"])
    y_test = np.array(test_batch["fine_labels"])
    
    # Create Y test wise matrix
    m = x_test.shape[0]
    c = (max(y_test) + 1) ## number of categories
    y_matrix_test = np.zeros((m,c))
    y_matrix_test[(range(m), y_test.astype(int))] = 1
    
    # Normalize training X
    x_mean = np.mean(x_train, axis=0)
    x_stdev = np.std(x_train, axis=0)
    for i, k in enumerate(x_stdev):
        if k == 0:
            x_stdev[i] = 1
    x_train = (x_train - x_mean) / x_stdev
    x_test = (x_test - x_mean) / x_stdev
    
    return x_train, x_test, y_train, y_test, y_matrix_train, y_matrix_test

# Initialization and Running the Model

def mlp_initialize(layer_widths, phis):
  """
  Use Numpy's random package to initialize a list of weights,
  a list of biases, and a list of activation functions for
  the number of units per layer provided in the argument.
  To pass the tests you will need to initialize the matrices
  in the following order:
  ws1, bs1, ws2, bs2, ..., wsl, bsl.
  :param layer_widths: a list of layer widths
  :return: a list of weights, a list of biases, and a list of
  phis, one for each layer
  """
  l = len(layer_widths)
  ws = []
  bs = []
  for k in range(l - 1):
    ws.append(np.random.normal(0, 0.1, (layer_widths[k], layer_widths[k + 1])))
    bs.append(np.abs(np.random.normal(0, 0.1, (1, layer_widths[k + 1]))))
  
  # Append softmax for output activation
  phis.append(mlp_softmax)
  return [ws, bs, phis]

## Running the Model

def mlp_run_mnist_GD(n_iter, u_layers, phis, alpha, eta_func, eta_details):
  """
  Prepare the MNIST data from the local directory and run desired optimization- 
  based MLP for the specified number of iterations, with the specified 
  units per layer, learning schedule, and regularization hyperparameter, 
  to estimate the parameters on the training data.

  :param n_iter: the number of iterations of training
  :param u_layers: array of units per layer
  :param alpha: regularization hyperparameter
  :param eta_func: learning rate schedule
  :param eta_details: learning rate schedule details
  :return: x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test,
  ws0, bs0, ws_hat, bs_hat, train_acc, test_acc
  """
  
  # Prepare Data
  x_train, x_test, y_train, y_test, y_matrix_train, y_matrix_test = mlp_MNIST_data_prep()

  # Initialize MLP
  ws0, bs0, phis = mlp_initialize(u_layers, phis) 

  # Train with Gradient Decent
  ws_hat, bs_hat, costs = mlp_gradient_descent(x_train, y_matrix_train, ws0, bs0, phis, alpha, n_iter, eta_func, eta_details)

  # Calculate Final training and generalization errors
  y_hat_train = mlp_predict(x_train, ws_hat, bs_hat, phis) 
  y_hat_test = mlp_predict(x_test, ws_hat, bs_hat, phis) 
  train_err = 1-(y_hat_train == y_train).mean() 
  test_err = 1-(y_hat_test == y_test).mean() 
  
  return [x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test, ws0, bs0, ws_hat, bs_hat, train_err, test_err, costs]

def mlp_run_mnist_SGD(n_iter, minibatches, u_layers, phis, alpha, eta_func, eta_details):
  """
  Prepare the MNIST data from the local directory and run desired optimization- 
  based MLP for the specified number of iterations, with the specified 
  units per layer, learning schedule, and regularization hyperparameter, 
  to estimate the parameters on the training data.

  :param n_iter: the number of iterations of training
  :param minibatches: minibatch size
  :param u_layers: array of units per layer
  :param alpha: regularization hyperparameter
  :param eta_func: learning rate schedule
  :param eta_details: learning rate schedule details
  :return: x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test,
  ws0, bs0, ws_hat, bs_hat, train_acc, test_acc
  """
  
  # Prepare Data
  x_train, x_test, y_train, y_test, y_matrix_train, y_matrix_test = mlp_MNIST_data_prep()

  # Initialize MLP
  ws0, bs0, phis = mlp_initialize(u_layers, phis) 

  # Train with SGD
  ws_hat, bs_hat, costs = mlp_SGD(x_train, y_matrix_train, minibatches, ws0, bs0, phis, alpha, n_iter, eta_func, eta_details)

  # Calculate Final training and generalization errors
  y_hat_train = mlp_predict(x_train, ws_hat, bs_hat, phis) 
  y_hat_test = mlp_predict(x_test, ws_hat, bs_hat, phis) 
  train_err = 1-(y_hat_train == y_train).mean() 
  test_err = 1-(y_hat_test == y_test).mean() 
  
  return [x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test, ws0, bs0, ws_hat, bs_hat, train_err, test_err, costs]

def mlp_run_mnist_SGD_results(n_iter, minibatches, u_layers, phis, alpha, eta_func, eta_details):
  """
  Prepare the MNIST data from the local directory and run desired optimization- 
  based MLP for the specified number of iterations, with the specified 
  units per layer, learning schedule, and regularization hyperparameter, 
  to estimate the parameters on the training data.

  :param n_iter: the number of iterations of training
  :param minibatches: minibatch size
  :param u_layers: array of units per layer
  :param alpha: regularization hyperparameter
  :param eta_func: learning rate schedule
  :param eta_details: learning rate schedule details
  :return: x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test,
  ws0, bs0, ws_hat, bs_hat, train_acc, test_acc
  """
  
  # Prepare Data
  x_train, x_test, y_train, y_test, y_matrix_train, y_matrix_test = mlp_MNIST_data_prep()

  # Initialize MLP
  ws0, bs0, phis = mlp_initialize(u_layers, phis) 

  # Train with SGD
  ws_hat, bs_hat, costs = mlp_SGD_results(x_train, y_train, x_test, y_test, y_matrix_train, minibatches, ws0, bs0, phis, alpha, n_iter, eta_func, eta_details)

  # Calculate Final training and generalization errors
  y_hat_train = mlp_predict(x_train, ws_hat, bs_hat, phis) 
  y_hat_test = mlp_predict(x_test, ws_hat, bs_hat, phis) 
  train_err = 1-(y_hat_train == y_train).mean() 
  test_err = 1-(y_hat_test == y_test).mean() 
  
  return [x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test, ws0, bs0, ws_hat, bs_hat, train_err, test_err, costs]

def mlp_run_CIFAR_10_GD(n_iter, u_layers, phis, alpha, eta_func, eta_details):
  """
  Prepare the CIFAR-10 data from the local directory and run desired optimization- 
  based MLP for the specified number of iterations, with the specified 
  units per layer, learning schedule, and regularization hyperparameter, 
  to estimate the parameters on the training data.

  :param n_iter: the number of iterations of training
  :param u_layers: array of units per layer
  :param alpha: regularization hyperparameter
  :param eta_func: learning rate schedule
  :param eta_details: learning rate schedule details
  :return: x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test,
  ws0, bs0, ws_hat, bs_hat, train_acc, test_acc
  """
  
  # Prepare Data
  x_train, x_test, y_train, y_test, y_matrix_train, y_matrix_test = mlp_CIFAR_10_data_prep()

  # Initialize MLP
  ws0, bs0, phis = mlp_initialize(u_layers, phis) 

  # Train with Gradient Decent
  ws_hat, bs_hat, costs = mlp_gradient_descent(x_train, y_matrix_train, ws0, bs0, phis, alpha, n_iter, eta_func, eta_details)

  # Calculate Final training and generalization errors
  y_hat_train = mlp_predict(x_train, ws_hat, bs_hat, phis) 
  y_hat_test = mlp_predict(x_test, ws_hat, bs_hat, phis) 
  train_err = 1-(y_hat_train == y_train).mean() 
  test_err = 1-(y_hat_test == y_test).mean() 
  
  return [x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test, ws0, bs0, ws_hat, bs_hat, train_err, test_err, costs]

def mlp_run_CIFAR_10_SGD(n_iter, minibatches, u_layers, phis, alpha, eta_func, eta_details):
  """
  Prepare the CIFAR-10 data from the local directory and run desired optimization- 
  based MLP for the specified number of iterations, with the specified 
  units per layer, learning schedule, and regularization hyperparameter, 
  to estimate the parameters on the training data.

  :param n_iter: the number of iterations of training
  :param minibatches: minibatch size
  :param u_layers: array of units per layer
  :param alpha: regularization hyperparameter
  :param eta_func: learning rate schedule
  :param eta_details: learning rate schedule details
  :return: x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test,
  ws0, bs0, ws_hat, bs_hat, train_acc, test_acc
  """
  
  # Prepare Data
  x_train, x_test, y_train, y_test, y_matrix_train, y_matrix_test = mlp_CIFAR_10_data_prep()

  # Initialize MLP
  ws0, bs0, phis = mlp_initialize(u_layers, phis) 

  # Train with SGD
  ws_hat, bs_hat, costs = mlp_SGD(x_train, y_matrix_train, minibatches, ws0, bs0, phis, alpha, n_iter, eta_func, eta_details)

  # Calculate Final training and generalization errors
  y_hat_train = mlp_predict(x_train, ws_hat, bs_hat, phis) 
  y_hat_test = mlp_predict(x_test, ws_hat, bs_hat, phis) 
  train_err = 1-(y_hat_train == y_train).mean() 
  test_err = 1-(y_hat_test == y_test).mean() 
  
  return [x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test, ws0, bs0, ws_hat, bs_hat, train_err, test_err, costs]

def mlp_run_CIFAR_10_SGD_results(n_iter, minibatches, u_layers, phis, alpha, eta_func, eta_details):
  """
  Prepare the CIFAR-10 data from the local directory and run desired optimization- 
  based MLP for the specified number of iterations, with the specified 
  units per layer, learning schedule, and regularization hyperparameter, 
  to estimate the parameters on the training data.

  :param n_iter: the number of iterations of training
  :param minibatches: minibatch size
  :param u_layers: array of units per layer
  :param alpha: regularization hyperparameter
  :param eta_func: learning rate schedule
  :param eta_details: learning rate schedule details
  :return: x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test,
  ws0, bs0, ws_hat, bs_hat, train_acc, test_acc
  """
  
  # Prepare Data
  x_train, x_test, y_train, y_test, y_matrix_train, y_matrix_test = mlp_CIFAR_10_data_prep()

  # Initialize MLP
  ws0, bs0, phis = mlp_initialize(u_layers, phis) 

  # Train with SGD
  ws_hat, bs_hat, costs = mlp_SGD_results(x_train, y_train, x_test, y_test, y_matrix_train, minibatches, ws0, bs0, phis, alpha, n_iter, eta_func, eta_details)

  # Calculate Final training and generalization errors
  y_hat_train = mlp_predict(x_train, ws_hat, bs_hat, phis) 
  y_hat_test = mlp_predict(x_test, ws_hat, bs_hat, phis) 
  train_err = 1-(y_hat_train == y_train).mean() 
  test_err = 1-(y_hat_test == y_test).mean() 
  
  return [x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test, ws0, bs0, ws_hat, bs_hat, train_err, test_err, costs]

def mlp_run_CIFAR_10_GD(n_iter, u_layers, phis, alpha, eta_func, eta_details):
  """
  Prepare the CIFAR-100 data from the local directory and run desired optimization- 
  based MLP for the specified number of iterations, with the specified 
  units per layer, learning schedule, and regularization hyperparameter, 
  to estimate the parameters on the training data.

  :param n_iter: the number of iterations of training
  :param u_layers: array of units per layer
  :param alpha: regularization hyperparameter
  :param eta_func: learning rate schedule
  :param eta_details: learning rate schedule details
  :return: x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test,
  ws0, bs0, ws_hat, bs_hat, train_acc, test_acc
  """
  
  # Prepare Data
  x_train, x_test, y_train, y_test, y_matrix_train, y_matrix_test = mlp_CIFAR_100_data_prep()

  # Initialize MLP
  ws0, bs0, phis = mlp_initialize(u_layers, phis) 

  # Train with Gradient Decent
  ws_hat, bs_hat, costs = mlp_gradient_descent(x_train, y_matrix_train, ws0, bs0, phis, alpha, n_iter, eta_func, eta_details)

  # Calculate Final training and generalization errors
  y_hat_train = mlp_predict(x_train, ws_hat, bs_hat, phis) 
  y_hat_test = mlp_predict(x_test, ws_hat, bs_hat, phis) 
  train_err = 1-(y_hat_train == y_train).mean() 
  test_err = 1-(y_hat_test == y_test).mean() 
  
  return [x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test, ws0, bs0, ws_hat, bs_hat, train_err, test_err, costs]

def mlp_run_CIFAR_100_SGD(n_iter, minibatches, u_layers, phis, alpha, eta_func, eta_details):
  """
  Prepare the CIFAR-100 data from the local directory and run desired optimization- 
  based MLP for the specified number of iterations, with the specified 
  units per layer, learning schedule, and regularization hyperparameter, 
  to estimate the parameters on the training data.

  :param n_iter: the number of iterations of training
  :param minibatches: minibatch size
  :param u_layers: array of units per layer
  :param alpha: regularization hyperparameter
  :param eta_func: learning rate schedule
  :param eta_details: learning rate schedule details
  :return: x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test,
  ws0, bs0, ws_hat, bs_hat, train_acc, test_acc
  """
  
  # Prepare Data
  x_train, x_test, y_train, y_test, y_matrix_train, y_matrix_test = mlp_CIFAR_100_data_prep()

  # Initialize MLP
  ws0, bs0, phis = mlp_initialize(u_layers, phis) 

  # Train with SGD
  ws_hat, bs_hat, costs = mlp_SGD(x_train, y_matrix_train, minibatches, ws0, bs0, phis, alpha, n_iter, eta_func, eta_details)

  # Calculate Final training and generalization errors
  y_hat_train = mlp_predict(x_train, ws_hat, bs_hat, phis) 
  y_hat_test = mlp_predict(x_test, ws_hat, bs_hat, phis) 
  train_err = 1-(y_hat_train == y_train).mean() 
  test_err = 1-(y_hat_test == y_test).mean() 
  
  return [x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test, ws0, bs0, ws_hat, bs_hat, train_err, test_err, costs]

def mlp_run_CIFAR_100_SGD_results(n_iter, minibatches, u_layers, phis, alpha, eta_func, eta_details):
  """
  Prepare the CIFAR-100 data from the local directory and run desired optimization- 
  based MLP for the specified number of iterations, with the specified 
  units per layer, learning schedule, and regularization hyperparameter, 
  to estimate the parameters on the training data.

  :param n_iter: the number of iterations of training
  :param minibatches: minibatch size
  :param u_layers: array of units per layer
  :param alpha: regularization hyperparameter
  :param eta_func: learning rate schedule
  :param eta_details: learning rate schedule details
  :return: x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test,
  ws0, bs0, ws_hat, bs_hat, train_acc, test_acc
  """
  
  # Prepare Data
  x_train, x_test, y_train, y_test, y_matrix_train, y_matrix_test = mlp_CIFAR_100_data_prep()

  # Initialize MLP
  ws0, bs0, phis = mlp_initialize(u_layers, phis) 

  # Train with SGD
  ws_hat, bs_hat, costs = mlp_SGD_results(x_train, y_train, x_test, y_test, y_matrix_train, minibatches, ws0, bs0, phis, alpha, n_iter, eta_func, eta_details)

  # Calculate Final training and generalization errors
  y_hat_train = mlp_predict(x_train, ws_hat, bs_hat, phis) 
  y_hat_test = mlp_predict(x_test, ws_hat, bs_hat, phis) 
  train_err = 1-(y_hat_train == y_train).mean() 
  test_err = 1-(y_hat_test == y_test).mean() 
  
  return [x_train, y_matrix_train, y_train, x_test, y_matrix_test, y_test, ws0, bs0, ws_hat, bs_hat, train_err, test_err, costs]
