# N_Pendulum
Python script to calculate the path of an arbitrary N-pendulum from initial conditions, then generate and animate images as a gif.

Requires Python modules: matplotlib, numpy, scipy, numba

Requires ffmpeg

# Brief Explanation

### Calculation
First use a Lagrangian-derived matrix to compute the acceleration of each bob given each bob's position, velocity. 

The acceleration matrix is then fed into `scipy.odeint()` for each timestep.

### Drawing
Drawing is done with matplotlib in `make_plot()`. Draws one frame every `N` timesteps to achieve the target FPS. 

Frames are drawn and temporarily saved to a folder `main_output` to be animated and deleted later.

### Animating
Animation is done with ffmpeg. After animation completes, all temporary frames are deleted from `main_output`, leaving behind the gif.


# Example Output
![4 Bob Example](https://github.com/evanm1455/N_Pendulum/blob/master/pend_30s_4bob.gif)
