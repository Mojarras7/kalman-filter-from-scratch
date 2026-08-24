import kalmanFilter
import numpy as np
import matplotlib.pyplot as plt
"""
Test for extendedKalman with simple non-linear equations. In this test we will only plot position in x
but this can be adjusted to plot y and z position as well. In this case i wanted to use sympy 
wich is a library for symbolic mathematics in Python, so we can define the state vector and the functions that describe the system with symbols.
"""

def test():

    state_vector = np.array([['x'], # the symbols of the state vector the detivative is going to be calculated from these symbols
                         ['y']])
    
    functions = np.array([["cos(x)"], # these are the equations of the system that describe the movement.
                           ["sin(y)"]])

    initial_E = np.eye(2) # the initial covariance matrix

    # the best values are R = 0.001 and Q = 5.0
    R = np.eye(2) * 0.2 # Process noise covariance matrix
    Q = np.eye(2) * 5.0 # Measurement noise covariance matrix

    H = np.array([[1,0],
                    [0,1]]) # this is meassurement matrix in this case our sensor is linear and we can measure the position directly, so we can use the identity matrix as the measurement matrix.

    steps = 41 # Define the number of steps you want to see
    n = np.arange(0, steps, 0.1)

    kalman = kalmanFilter.extendedKalman(state_vector, functions,initial_E, H, R, Q)
    pos = []
    sensor_pos_x = []
    kalman_pose = []

    for x in n:
        px = np.cos(x)
        pos.append(px)

        # we now add noise
        sensor_x = np.random.normal(px, 0.3) # simulate a sensor reading with noise with a standard deviation of 0.3
        sensor_pos_x.append(sensor_x)

        noisy_pos = np.array([[sensor_x],
                              [0]])
        
        current_states = np.array([[x], # the current state of the system is the current value of x and y, in this case we are only interested in x but we can also use y if we want to see the movement in y as well.
                                  [x]])
        #Filter the sensor
        kalman.filter(current_states, noisy_pos) # we pass the current state and the noisy position to the filter function of the extended Kalman filter, this function will calculate the new state and covariance matrix based on the current state and the noisy position.
        kalman_pose.append(kalman.x[0])


    # Plot the results
    plt.plot(n, pos, label ='True Position', color ='blue', linestyle='dashed')
    plt.plot(n, sensor_pos_x, label ='Noisy Position', color='lightgreen')
    plt.plot(n, kalman_pose, label='Kalman Filtered X', color='red', linestyle='dashed')

    plt.grid()
    plt.legend()
    plt.show() 

if __name__ == '__main__':
    test()
   