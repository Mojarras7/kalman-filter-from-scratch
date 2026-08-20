import kalmanFilter
import numpy as np
import matplotlib.pyplot as plt

def test():

    state_vector = np.array([['x'],
                         ['y']])
    
    functions = np.array([["cos(x)"],
                           ["sin(y)"]])

    initial_E = np.eye(2)

    # the best values are R = 0.001 and Q = 5.0
    R = np.eye(2) * 0.2
    Q = np.eye(2) * 5.0

    H = np.array([[1,0],
                    [0,1]]) 

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
        sensor_x = np.random.normal(px, 0.3)
        sensor_pos_x.append(sensor_x)

        noisy_pos = np.array([[sensor_x],
                              [0]])
        
        current_states = np.array([[x],
                                  [x]])
        #Filter the sensor
        kalman.filter(current_states, noisy_pos)
        kalman_pose.append(kalman.x[0])


    # print(pos)
    plt.plot(n, pos, label ='True Position', color ='blue', linestyle='dashed')
    plt.plot(n, sensor_pos_x, label ='Noisy Position', color='lightgreen')
    plt.plot(n, kalman_pose, label='Kalman Filtered X', color='red', linestyle='dashed')

    plt.grid()
    plt.legend()
    plt.show() 

if __name__ == '__main__':
    test()
   