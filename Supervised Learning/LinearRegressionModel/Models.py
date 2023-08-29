class LinearRegression:
    
    # Initialization function: takes two parameters features, target values
    def __init__(self, x, y):
        if len(x) != len(y):
            raise Exception("Length of features and target values are not equal")
        self.x = x
        self.y = y
        self.m = len(x)

    # Function to calculate the cost
    def compute_cost(self, w, b):
        cost = 0
        
        for i in range(self.m):
            f_wb = w * self.x[i] + b
            cost = cost + (f_wb - self.y[i])**2
        total_cost = 1 / (2 * self.m) * cost

        return total_cost
    
    # Function to calculate gradient
    def compute_gradient(self, w, b):
        dj_dw = 0
        dj_db = 0
        
        for i in range(self.m):  
            f_wb = w * self.x[i] + b 
            dj_dw_i = (f_wb - self.y[i]) * self.x[i] 
            dj_db_i = f_wb - self.y[i] 
            dj_db += dj_db_i
            dj_dw += dj_dw_i 
        dj_dw = dj_dw / self.m 
        dj_db = dj_db / self.m
            
        return dj_dw, dj_db
    
    # Gradient Descent Function
    def gradient_descent(self, w_in, b_in, alpha, num_iters): 
        # An array to store cost J and w's at each iteration primarily for graphing later
        J_history = []
        p_history = []
        b = b_in
        w = w_in
        
        for i in range(num_iters):
            # Calculate the gradient and update the parameters using gradient_function
            dj_dw, dj_db = self.compute_gradient(w , b)     
            b = b - alpha * dj_db
            w = w - alpha * dj_dw             

            # Save cost J at each iteration
            if i<100000:      # prevent resource exhaustion 
                J_history.append( self.compute_cost(w , b))
                p_history.append([w,b])
            if i % (num_iters/10) == 0:
                print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                    f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                    f"w: {w: 0.3e}, b:{b: 0.5e}")
    
        return w, b