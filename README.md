# Deep Lagrange interpolation and Semi Lagrangian solver


In this project, we aim to solve transport equations using the Semi-Lagrangian implementing a scheme and analyze its convergence and accuracy compared to the exact solution. 
This project is supervised by Emmanuel Franck and Laurent Navoret from the French National Institute for Research in Computer Science and Control (INRIA) and the University of Strasbourg.

## Objectives

The primary goal of this project is to enhance the accuracy of the current resolution method by incorporating deep learning techniques while maintaining convergence properties. We will implement a Semi-Lagrangian scheme then we will apply deep interpolation techniques and compare the results with traditional methods.


## Theorical Framework

We focus on solving the transport equation with a given velocity and initial condition.

$$
\begin{cases}
\partial_t u + a \partial_x u = 0 \\
u(t=0, x) = u_0(x, \mu)
\end{cases}
$$

### Semi-Lagrangian Scheme
The semi-Lagrangian method is a numerical technique that tracks fluid particles through time and space. We will use this scheme to approximate the solution by interpolating the values from the previous time step.

We implemented this scheme with the Lagrange interpolation and the Deep Lagrange interpolation and with the function:

$$
f(x) = exp(-(x - 0.5)^2 / 0.1^2)
$$

## Results
We first calculated the solution using the lagrange interpolation operator on a small mesh with nx = 40, nt = 50, a=1 and we obtained:

![ezgif com-video-to-gif](https://github.com/master-csmi/2023-m1-inria-disl/assets/87640597/d9fcf34e-f4e0-41d5-94ef-31c79c4f7d2d)

And the error that increases at each time step:

![errors1](https://github.com/master-csmi/2023-m1-inria-disl/assets/87640597/07dcbf0f-17aa-49fe-9a47-3639da96327b)

Adding more points nx_bis= 100, nt_bis = 200, a_bis = 1 we obtained:

![ezgif com-video-to-gif-5](https://github.com/master-csmi/2023-m1-inria-disl/assets/87640597/10bf96ba-88ea-4f23-aae8-a2f28dc5451d)

Now implementing the Deep Lagrange interpolator we obtained:

![ezgif com-video-to-gif-3](https://github.com/master-csmi/2023-m1-inria-disl/assets/87640597/b262d037-4e62-496d-afad-4b4c463782d6)

Here we can see that the solution u_deep is closer to our exact solution, because the error is relatively small:

![errors2](https://github.com/master-csmi/2023-m1-inria-disl/assets/87640597/ff0e047e-29fe-4971-8eac-51c60e6a8a5f)

Adding a small perturbation to our exact solution we get:

![ezgif com-video-to-gif-4](https://github.com/master-csmi/2023-m1-inria-disl/assets/87640597/e670be80-8c35-48f9-90bb-4c3b8d9ffec0)

As expected the error has grown and our solution is less accurate:

![errors3](https://github.com/master-csmi/2023-m1-inria-disl/assets/87640597/f90716ca-8a7d-4c28-8bc7-e10aa596e7c1)

For the different measurement of convergence we obtained:

### In space

![conv1](https://github.com/master-csmi/2023-m1-inria-disl/assets/87640597/9c2cfee5-1314-4560-b05a-e48da6173c88)

![conv2](https://github.com/master-csmi/2023-m1-inria-disl/assets/87640597/472031b2-465e-4807-9306-f13fe1450370)

## In time

![conv3](https://github.com/master-csmi/2023-m1-inria-disl/assets/87640597/ab18a405-fc09-440f-8519-02cd12b68ef5)

## Conclusions

Our findings demonstrate that the Semi-Lagrangian scheme combined with the deep interpolation operator shows promise in improving the accuracy of transport equation solutions. However, there is still further research to be done, with implementation of the deep Lagrange interpolation with PINNs to optimize the deep learning models and explore their full potential in solving more complex transport equations.


