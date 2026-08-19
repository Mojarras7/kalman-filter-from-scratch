import kalmanFilter
import numpy as np
def test():
    extendedKalman =  kalmanFilter.extendedKalman()

    state_vector = np.array([['x'],
                         ['y']])
    
    functions = np.array([["sin(x) + 2*x"],
                           ["cos(y) + x**2"]])

    prev_E = np.array([[0]])


    extendedKalman.prediction(state_vector, functions, prev_E)