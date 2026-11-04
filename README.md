# Kalman Filter from Scratch

## Overview

The Kalman Filter is a mathematical algorithm that is used to work with uncertainties in measurements and predictions. This algorithm combines probability and linear algebra. It is widely used in various fields such as robotics, computer vision, and control systems.

This repository contains the implementation of a Kalman Filter from scratch using NumPy and SymPy. The project includes two main classes:

- `simpleKalman`: for linear systems.
- `extendedKalman`: for non-linear systems using Jacobian linearization.

## Example and usage

The script is designed to be modular, you can easily use the classes in other projects, but running the script will run the main function which contains a simple example of the kalman filter in action in a MRUA scenario. With Gaussian noise added to the measurements, the Kalman filter is able to estimate the true position of the object, even when the measurements are extremely noisy. Matplotlib is used to visualize the results only in the main function, but it is not a requirement to use the classes.

![Simple Kalman Filter](assets/simple_kalman_filter.png)

## Extended Kalman Filter

The extended kalman filter in this case uses the sympy library to calculate the jacobian from the symbolic representation of the functions. I used my own implementation of the jacobian matrix calculation using sympy library, but you can also use `sympy.jacobian()` to calculate the jacobian matrix. My function returns the evaluated jacobian matrix with respect to the current state vector. Also will show the symbolic jacobian matrix.

![Extended Kalman Filter](assets/extended_kalman_filter.png)

## Running the Examples

Run both examples:

```bash
python kalmanFilter.py
```

Or run each test individually:

```bash
python testSimple.py
python testExtended.py
```

## Reference

The equations of this algorithm are widely known, but my implementation is based on the book *Probabilistic Robotics* by Sebastian Thrun, Wolfram Burgard, and Dieter Fox.
