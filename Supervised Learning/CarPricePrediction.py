import numpy as np
from Model import LinearRegression

# Import data from csv file.
X = np.genfromtxt('CarData.csv', delimiter=',', dtype=float)
# Second row was for prices in the data set so I slice it and paste it into Y values.
Y = X[:, 1]
X = np.delete(X, (1), axis=1)
"""
X (ndarray(3000, 5)):

            Year    Transmission    Mileage     Fuel Type       Engine Size
x^1:    [   2016        -1           20706          0               1.2    ]
x^2:    [   2017        -1           15979          0               1.2    ]
x^3:    [   2017         1           11291          0               1.2    ]
...    
x^2998: [   2012        -1           40991          0               1.2    ]
x^2999: [   2017        -1           25407          0               1.2    ]
x^3000: [   2016         1           13871          0               1.0    ]

Transmission:
(-1 ): "Manuel"
( 0 ): "Semi-Automatic"
( 1 ): "Automatic"

Fuel Type:
( 0 ): "Petrol"
( 1 ): "Diesel"
"""

# We construct a model using the class we made.
carPriceModel = LinearRegression(X,Y) 

# We are scaling values using the module we made.
carPriceModel.X = LinearRegression.standardize(carPriceModel.X)

# Parameters
w_init = np.zeros(5)
b_init = 0.
iterations = 100
alpha = 1.0

# We run the gradient descent algorithm we made.
w_final, b_final = carPriceModel.gradient_descent(w_init, b_init, alpha, iterations)
print(f"b , w found by gradient descent: {b_final:0.2f} , {w_final} ")

# For testing purposes we can estimate price for data outside of training set.
test_X = np.array([ [2019,  1,  4789,   0,  2.0],
                    [2015,  0, 20469,   0,  1.2],
                    [2013,  0, 15352,   0,  1.4],
                    [2017,  0, 23656,   0,  1.2],
                    [2014,  0, 40736,   1,  1.4]])

# Prices
test_Y = np.array([18480, 9279, 7850, 10695, 8229]) 

# We are scaling values using the module we made.
test_X = LinearRegression.standardize(test_X)

for i in range(len(test_X)):
    f_wb = np.dot(test_X[i], w_final) + b_final
    print(f"For test({i+1}): Model Estimation: {f_wb:0.0f}£ Real Price: {test_Y[i]}£")

