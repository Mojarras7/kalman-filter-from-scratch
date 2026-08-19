"""
A kalman filter implementation from scratch from simple kalman all the way to extended kalman filter.
"""

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


class extendedKalman(simpleKalman):
    def __init__(self, A, initial_x, B, u, initial_E, C, R, Q):
        super().__init__(A, initial_x, B, u, initial_E, C, R, Q)
    # to do


def main():

    # Test for simpleKalman with simple MRUA equations
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
                    [0,0,1,0,0,0]]) # 

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

    # Plot the x position 
    import matplotlib.pyplot as plt

    pos = []
    noisy_pos_x = []
    noisy_pos_y = []
    noisy_pos_z = []
    kalman_pose = []

    #initial state
    x = initial_x

    # plot n steps
    n = 51 # change depending of how much steps you want

    kalman = simpleKalman(A, initial_x, B, u, initial_E, C, R, Q)

    for _ in range(n):
        x = A@x + B@u
        # print(f"Step {n} : {x}")
        pos.append(x[0])

        # we now add noise 
        noise_x = abs(np.random.normal(x[0], 0.2))
        noisy_pos_x.append(noise_x)


        noisy_pos = np.array([noise_x,
                              [0],
                              [0]])

        #kalman filer test
        kalman.filter(noisy_pos)
        kalman_pose.append(kalman.x[0])




    # print(kalman_pose, end=" ")
    print(noisy_pos)

    plt.plot(np.arange(0,n,1), pos, label='True X', color='green', linestyle='dashed')
    plt.plot( np.arange(0,n,1), noisy_pos_x, label='Noisy X', color='blue')
    plt.plot(np.arange(0,n,1), kalman_pose, label='Kalman Filtered X', color='red')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()