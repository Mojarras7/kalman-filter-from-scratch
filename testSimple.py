import kalmanFilter
import numpy as np
import matplotlib.pyplot as plt
"""
Test for simpleKalman with simple MRUA equations. In this test we will only plot position in x
but this can be adjusted to plot y and z position as well. 
"""

def test():
    dt = 0.1
    A = np.array([[1,0,0,dt,0,0],
                    [0,1,0,0,dt,0],
                    [0,0,1,0,0,dt],
                    [0,0,0,1,0,0],
                    [0,0,0,0,1,0],
                    [0,0,0,0,0,1]])
    
    B = np.array([[0.5*dt**2,0,0],
                    [0,0.5*dt**2,0],
                    [0,0,0.5*dt**2],
                    [dt,0,0],
                    [0,dt,0],
                    [0,0,dt]])
    
    C = np.array([[1,0,0,0,0,0],
                    [0,1,0,0,0,0],
                    [0,0,1,0,0,0]])  

    # Process noise covariance matrix R and measurement noise covariance matrix Q
    R = np.eye(6) * 0.001 # Process noise covariance matrix
    Q = np.eye(3) * 5.0 # Measurement noise covariance matrix

    # Initial state and covariance
    initial_x = np.array([[2.0], # x position 
                            [0], # y position
                            [0], # z position
                            [0.1], # x velocity
                            [0], # y velocity
                            [0]]) # z velocity
    
    initial_E = np.array([[1,0,0,0,0,0],
                            [0,1,0,0,0,0],
                            [0,0,1,0,0,0],
                            [0,0,0,1,0,0],
                            [0,0,0,0,1,0],
                            [0,0,0,0,0,1]])

    u  = np.array([[0.3],
                    [0],
                    [0]])



    pos = []
    noisy_pos_x = []
    kalman_pose = []

    #initial state
    x = initial_x

    # plot n steps
    n = 51 # change depending of how much steps you want

    kalman = kalmanFilter.simpleKalman(A, initial_x, B, u, initial_E, C, R, Q) # Create a new instance of the Kalman filter
    for _ in range(n): # this is the main loop where we will simulate the movement of the object and apply the Kalman filter
        x = A@x + B@u
        # print(f"Step {n} : {x}")
        pos.append(x[0]) # Append the true x position to the list

        # we now add noise 
        noise_x = np.random.normal(x[0], 0.2)
        noisy_pos_x.append(noise_x) # we create an array with the noisy position to simulate a sensor reading

        noisy_pos = np.array([noise_x,
                                [0],
                                [0]])

        #kalman filer test
        kalman.filter(noisy_pos) # now we apply the Kalman filter to the noisy position
        kalman_pose.append(kalman.x[0])

    # print(kalman_pose, end=" ")
    # print(noisy_pos)

    # plot the results
    plt.plot(np.arange(0,n,1), pos, label='True X', color='blue', linestyle='dashed')
    plt.plot( np.arange(0,n,1), noisy_pos_x, label='Noisy X', color='lightgreen')
    plt.plot(np.arange(0,n,1), kalman_pose, label='Kalman Filtered X', color='red')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test()