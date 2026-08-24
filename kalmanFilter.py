"""
A kalman filter implementation from scratch from simple kalman all the way to extended kalman filter.
"""

import sympy as sp
import numpy as np


class simpleKalman():
    def __init__(self, A, initial_x, B, u, initial_E, C, R, Q):
        self.A = A
        self.B = B
        self.C = C
        self.R = R
        self.Q = Q
        self.u = u
        self.x = initial_x
        self.E = initial_E

    def filter(self, z):
        """
        The kalman filter is a two step process, prediction and update. 
        In the prediction step we use the previous state and covariance to predict the next state and covariance.
        In the update step we use the predicted state and covariance to update the state and covariance with the measurement.
        """
        
        predicted_x, predicted_cov = self.prediction(self.x, self.E)
        updated_x, updated_cov = self.update(predicted_x, predicted_cov, z)

        self.x = updated_x
        self.E = updated_cov

    def prediction(self, prev_x, prev_E):
        """
        from the equations: 
        μ = A*μ-1*+ B*u:
        Σ = A*Σ*AT + R

        Where:
        μ = the returned predicted state = x
        Σ = the returned predicted covariance = E
        μ-1 = the previouse state = prev_x
        u = the input vector = u
        A = the state transition matrix = A
        B = the control input matrix = B
        R = Process noise = R
        """
        
        predicted_x = self.A@prev_x + self.B@self.u
        predicted_cov = self.A@prev_E@self.A.transpose() + self.R 

        return predicted_x, predicted_cov

    def update(self, predicted_x, predicted_cov, z):
        """
        from the equations:
        K = Σ*CT*(C*Σ*CT + Q)^-1
        μ = μ + K*(z - C*μ) # here we use the predicted state as μ to calculate the updated state
        Σ = (I - K*C)*Σ # the same 


        Where:
        μ = the returned updated state = updated_x
        Σ = the returned updated covariance = updated_cov
        K = the kalman gain = K
        z = the measurement vector = z
        C = the measurement matrix = C
        Q = the measurement noise = Q
        """
        transC = self.C.transpose()
        K = predicted_cov@transC@np.linalg.inv(self.C@predicted_cov@transC + self.Q)

        updated_cov = predicted_cov - K@self.C@predicted_cov
        updated_x = predicted_x + K@(z - self.C@predicted_x)

        return updated_x, updated_cov


class extendedKalman():
    def __init__(self, state_vector,functions,initial_E, R, H, Q):
        self.state_vector = state_vector
        self.functions = functions
        self.E = initial_E
        self.R = R
        self.H = H
        self.Q = Q

        self.getEvaluated(self.state_vector, self.functions)
        
    def filter(self, current_states, z):
        """
        The main idea in the extended Kalman filter is to linearize the non-linear system.
        The extended kalman filter uses the linearization via Taylor expansion.
        We use matrix known as the Jacobian matrix to linearize the system.

        One example of a jacobian matrix is the following:
        f(x) = [x1^2 + x2^2, x1*x2]

        J = [2*x1, 2*x2]
            [x2, x1]

        Where: 
        f(x) = the non-linear function
        J = the jacobian matrix
        2*x1, 2*x2, x2, x1 = the partial derivatives of f(x) with respect to x1 and x2

        You can notice that the first row of the jacobian matrix is the partial derivatives of the first element of f(x) with respect to x1 and x2,
        and the second row is the partial derivatives of the second element of f(x) with respect to x1 and x2.
    
        """

        predicted_x, predicted_cov = self.prediction(current_states, self.E)
        updated_x, updated_cov = self.update(predicted_x, predicted_cov, z)

        self.x = updated_x
        self.E = updated_cov

    def getJacobian(self, state_vector, functions, current_states):
        """
        This function is used to calculate the jacobian matrix of the non-linear functions with respect to the state vector.
        I used my owm implementation of the jacobian matrix calculation using sympy library, but you can also use the sympy.jacobian() function to calculate the jacobian matrix.
        This funcition will return the evaluated jacobian matrix with respect to the current state vector. Also will show the symbolic jacobian matrix.
        """
        
        sym = [] # A list to store the symbols of the state vector
        exp = [] # A list to store the expressions of the functions

        for n in range(np.size(state_vector)): # Loop through the state vector wich is the same size as the functions vector
            symbol = sp.symbols(state_vector[n][0])# Create a symbol for each element of the state vector
            sym.append([symbol]) # append the symbol as a list to the sym list

            expresion = sp.sympify(functions[n][0])
            exp.append([expresion])


        arr_sym = np.array(sym)
        arr_exp= np.array(exp)

        # print(arr_sym)
        # print(arr_exp)

        
        row_jacobian_sym = [] # A list to store the rows of the jacobian matrix
        row_jacobian = []
        for e in range(np.size(arr_exp)): # we iterate for each expression
           derivatives_sym = [] # for each expression we create a list to store the partial derivatives
           derivatives = []
           for s in range(np.size(arr_sym)): # we iterate for each symbol we  need to iterate in order to get the partial derivative of each expression with respect to each symbol
                partial_derivative = sp.Derivative(arr_exp[e][0],arr_sym[s][0]) # we calculate the partial derivative of the expression with respect to the symbol
                partial_derivative = partial_derivative.doit()

                function_derivative = sp.lambdify(arr_sym[s][0], partial_derivative)
                derivatives.append(function_derivative)

                derivatives_sym.append(partial_derivative) # we add the calculated partial derivative to the list

           row_jacobian.append(derivatives)
           row_jacobian_sym.append(derivatives_sym) # the calculated derivatives are for the first function with respect to all the symbols.

        jacobian_sym = np.array(row_jacobian_sym) # at the end we convert the list to a numpy array to be able to use it in the kalman filter equations
        jacobian = np.array(row_jacobian)
        print(f"The Symbolical Jacobian is {jacobian_sym} " )
        print(f"The Jacobian is {jacobian}")

        # Now we evaluate the jacobian

        row_eval = []
        for e in range(np.size(arr_exp)):
            evaluated = []
            for s in range(np.size(arr_sym)):
                evaluated.append(jacobian[e][s](current_states[s][0]))
            row_eval.append(evaluated)

        eval_jacobian = np.array(row_eval)

        print(eval_jacobian)

        return eval_jacobian
                

    def getEvaluated(self,state_vector, functions):
        """
        This function is used to evaluate the functions of the system with respect to the state vector.
        We use the sympy library to create a symbolic representation of the functions and then we use the lambdify function to create a numerical representation of the functions.
        """
        self.f_eval = []
        for n in range(len(functions)):
            sym = sp.symbols(state_vector[n][0])
            expr = sp.sympify(functions[n][0])
            func_num = sp.lambdify(sym, expr, modules='numpy')
            self.f_eval.append(func_num)

    def prediction(self, current_states, prev_E):
        """
        The prediction step is the same as in the simple kalman filter, 
        but we need to linearize the non-linear functions using the Jacobian matrix.
        """
        self.G =  self.getJacobian(self.state_vector, self.functions, current_states)

        predicted_x = []
        for n in range(len(self.f_eval)):
            val = self.f_eval[n](current_states[n][0])
            predicted_x.append([val])

        predicted_x = np.array(predicted_x) 
        predicted_cov = self.G@prev_E@self.G.transpose() + self.R

        return predicted_x, predicted_cov

    def update(self, predicted_x, predicted_cov, z):
        """The update step is the same as the simple kalman"""
        transH = self.H.transpose()
        K = predicted_cov@transH@np.linalg.inv(self.H@predicted_cov@transH + self.Q)

        updated_cov = predicted_cov - K@self.H@predicted_cov
        updated_x = predicted_x + K@(z - self.H@predicted_x)

        return updated_x, updated_cov
     
def main():
    import testSimple # import testSimple file
    import testExtended # import testExtended file

    testSimple.test()
    testExtended.test()


if __name__ == "__main__":
    main()