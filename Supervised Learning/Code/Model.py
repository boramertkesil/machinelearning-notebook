import numpy as np
import math

class Regression:

    # Regression class is initialized by itself.
    def __init__(self, X, Y):
        """
        Args:
        X (ndarray (m,n)): Data, m examples with n features
        Y (ndarray (m)): Target values
        """
        # Raise exception if length of features is not equal to length of target values. 
        if len(X) != len(Y):
            raise Exception("Length of features and target values are not equal")
        self.X = X
        self.Y = Y
        self.m = len(X)

    def mean(arr):
        mean = np.mean(arr, axis=0)
        return mean

    def standard_deviation(arr):
        standard_deviation = np.std(arr, axis = 0)
        return standard_deviation
    
    def standardize(arr):
        standardized = (arr - Regression.mean(arr)) / Regression.standard_deviation(arr)
        return standardized


class LinearRegression(Regression):
    
    def cost(self, W, b):
        """
        Args:
        w (ndarray (n)): w values
        b (scalar): b value

        Returns:
        cost (scalar): cost value
        """
        f_Wb = self.X @ W + b
        error = f_Wb - self.Y
        cost = np.sum(error**2) / (2 * self.m)
        return cost

    def gradient(self, W, b):
        """
        Args:
        w (ndarray (n)): w values
        b (scalar): b value

        Returns:
        dj_dw (ndarray (n)): gradient values for all w
        dj_db (scalar): gradient value for b
        """
        error = (self.X @ W + b) - self.Y
        dj_dw = (error @ self.X) / self.m
        dj_db = np.sum(error) / self.m
        return dj_dw, dj_db
    
    def gradient_descent(self, W_init, b_init, alpha, num_iters):
        """
        Args:
        W_init (ndarray (n)): w values
        b_init (scalar): b value
        alpha (scalar): alpha
        num_iters (scalar): number of iterations

        Returns:
        W (ndarray (n)): final value of W
        b (scalar):  final value of b
        """
        W = W_init
        b = b_init

        for i in range(num_iters):

            # Calculate the gradient and update the parameters
            dj_dw, dj_db = self.gradient(W, b)#1x4

            W -= alpha * dj_dw
            b -= alpha * dj_db

            # Print cost every at intervals 10 times or as many iterations if < 10
            if i% math.ceil(num_iters / 10) == 0:
                print(f"Iteration {i:4d}: Cost {self.cost(W, b):8.2f} ")
        
        return W, b
    
class LogisticRegression(Regression):
        
    def sigmoid(self, W, b):
        z = self.X @ W + b
        g = 1 / (1 + np.exp(-z))
        return g
    
    def loss(self, W, b):
        loss = -self.Y * np.log(self.sigmoid(W, b)) - (1 - self.Y) * np.log(1 - self.sigmoid(W, b))
        return loss
    
    def cost(self, W, b):
        cost = np.sum(self.loss(W, b)) / self.m
        return cost
    
    def gradient(self, W, b): 
        """
        Args:
        w (ndarray (n)): model parameters  
        b (scalar)      : model parameter
        Returns
        dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
        dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
        """
        error = self.sigmoid(W, b) - self.Y
        dj_dw = (error @ self.X) / self.m
        dj_db = np.sum(error) / self.m

        return dj_dw, dj_db
    
    def gradient_descent(self, W_init, b_init, alpha, num_iters): 
        """
        Performs batch gradient descent
        
        Args:
        X (ndarray (m,n)   : Data, m examples with n features
        y (ndarray (m,))   : target values
        w_init (ndarray (n,)): Initial values of model parameters  
        b_init (scalar)      : Initial values of model parameter
        alpha (float)      : Learning rate
        num_iters (scalar) : number of iterations to run gradient descent
        
        Returns:
        W (ndarray (n)): final value of W
        b (scalar):  final value of b
        """

        W = W_init
        b = b_init
        
        for i in range(num_iters):
            # Calculate the gradient and update the parameters
            dj_dw, dj_db = self.gradient(W, b)

            # Update Parameters using w, b, alpha and gradient
            W = W - alpha * dj_dw               
            b = b - alpha * dj_db               

            # Print cost every at intervals 10 times or as many iterations if < 10
            if i% math.ceil(num_iters / 10) == 0:
                print(f"Iteration {i:4d}: Cost {self.cost(W, b)} ")
            
        return W, b