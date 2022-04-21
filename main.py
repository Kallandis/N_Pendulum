import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from itertools import combinations
import os
from time import perf_counter
from numba import jit


# jit compilation brings total calculation time for 20 bobs from 286s -> 135s
# total time = time_calculating + time_drawing + time_animating
# time_calculating: 161s -> ~10s
@jit
def theta_accel(theta, theta_dot):
    # Computes instantaneous acceleration of theta as fxn of theta, theta_dot
    # takes inputs from deriv, which holds only one row of data at a time
    A = np.zeros([N, N])
    B = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            for k in range(max(i, j), N):
                A[i][j] += m[k]
                B[i][j] += m[k]
            if i == j:
                A[i][j] *= l[j]
                B[i][j] *= g * np.sin(theta[i])
            else:
                A[i][j] *= l[j] * np.cos(theta[i] - theta[j])
                B[i][j] *= l[j] * theta_dot[j] ** 2 * np.sin(theta[i] - theta[j])

    C = np.matmul(np.linalg.inv(A), B)
    tdd = np.dot(C, np.full(N, -1))
    return tdd


def deriv(y, t):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta = y[0:N]
    theta_d = y[N:]
    tdd = theta_accel(theta, theta_d)
    return np.append(y[N:], tdd)


def calc_E(y):
    T = 0
    V = 0
    theta = y[0:N]
    theta_d = y[N:]

    # calculate T
    for i in range(N):
        temp = 0
        for j in range(i + 1):  # pure terms of the form: 0.5 * m * l^2 * td^2
            temp += (l[j] * theta_d[j])**2

        if i > 0:   # cross-terms of the form: m1 * l1 * l2 * td1 * td2 * cos(t1 - t2)
            combos = list(combinations(range(i+1), 2))
            for combo in combos:
                a, b = combo
                temp += 2 * l[a]*l[b] * theta_d[a]*theta_d[b] * np.cos(theta[a] - theta[b])

        temp *= 0.5 * m[i]
        T += temp

    # calculate V
    for i in range(N):
        temp = 0
        # V looks like: -(m1 + m2 + m3) * g * l1 * cos(t1) - (m2 + m3) * g * l2 * cos(t2) - m3 * g * l3 * cos(t3)
        for k in range(i, N):
            temp += m[k]
        temp *= -1 * g * l[i] * np.cos(theta[i])
        V += temp

    return T + V


def xy_coords(y):
    # calculate x, y for each bob from state vector "y"
    thetas = y.T[0:N]
    xs = np.zeros([N, timeSteps])
    ys = np.zeros([N, timeSteps])
    xs[0] = l[0] * np.sin(thetas[0, :])
    ys[0] = -l[0] * np.cos(thetas[0, :])
    for i in range(1, N):
        xs[i] = xs[i-1] + l[i] * np.sin(thetas[i, :])
        ys[i] = ys[i-1] - l[i] * np.cos(thetas[i, :])

    return xs, ys


def make_plot(xs, ys, i, r, maxTrail, maxFrames):
    # Plot and save an image of the double pendulum configuration for time point i.
    global imgFolder

    # x, y coords of each bob at frame i
    xcoords = [bob[i] for bob in xs]
    ycoords = [bob[i] for bob in ys]

    # draw the rods
    xrod = np.append(np.array([0]), xcoords)
    yrod = np.append(np.array([0]), ycoords)
    ax.plot(xrod, yrod, lw=0.5, c='k')

    # draw anchor point
    c0 = Circle((0, 0), r * 0.5, fc='k', zorder=10)
    ax.add_patch(c0)

    # draw bobs and fading trails behind them
    for k in range(N):
        circ = Circle((xs[k][i], ys[k][i]), r, fc=(k / N, 0.3, 0.3), ec=(k / N, 0.3, 0.3), zorder=10)
        ax.add_patch(circ)

        # The trail will be divided into ns segments and plotted as a fading line.
        ns = 20
        s = maxTrail // ns

        for j in range(ns):
            imin = i - (ns - j) * s
            if imin < 0:
                continue
            imax = imin + s + 1
            # The fading looks better if we square the fractional length along the trail
            alpha = (j / ns) ** 2
            ax.plot(xs[k][imin:imax], ys[k][imin:imax], c=(k / N, 0.3, 0.3), solid_capstyle='butt',
                    lw=2, alpha=alpha)

    # Center the image on the fixed anchor point, and ensure the axes are equal
    a = sum(l) + r
    ax.set_xlim(-a, a)
    ax.set_ylim(-a, a * 0.6)
    # plt.text(0, a / 2, 'FRAME: ' + str(i // di) + '   E: ' + '{:.2f}'.format(energies[i]), ha='center', fontsize='large')
    plt.text(0, a / 2, f'FRAME: {i // di} / {maxFrames}', ha='center', fontsize=25)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    # plt.savefig(imgFolder + '/_img{:04d}.png'.format(i // di))  # dpi=72
    plt.savefig(imgFolder + '/_img{:04d}.png'.format(i // di))  # dpi=72
    plt.cla()


if __name__ == "__main__":

    # set constants and initial conditions
    startTime = perf_counter()
    # imgFolder = __file__.split('\\')[-1][:-3] + 'Frames'
    imgFolder = "main_output"
    if not os.path.exists(imgFolder):
        os.mkdir(imgFolder)

    N = 20
    l = np.full(N, 1)
    m = np.full(N, 1)
    g = 9.81

    # Maximum time, time point spacings and the time grid (all in s).
    tmax, dt = 30, 0.01
    t = np.arange(0, tmax, dt)
    timeSteps = t.size

    # initial condition state vector. theta, theta_dot for each bob. must be 1D array for scipy.odeint()
    # N theta entries, N theta_dot entries = 2N total entries
    y0 = np.zeros(2 * N)
    for i in range(N):
        y0[i] = np.pi/1.8   # theta = 0 oriented along -y direction

    # Do the numerical integration of the equations of motion
    print('\nCALCULATING PATH\n')
    # y is an array of state vectors, iterated with each timestep dt
    y = odeint(deriv, y0, t)
    calcFinishTime = perf_counter()

    # check if energy is conserved
    checkE = True
    if checkE:
        EDRIFTMAX = 0.05
        # Total energy from the initial conditions
        initE = calc_E(y0)
        finalE = calc_E(y[-1])
        deltaE = finalE - initE
        if np.abs(deltaE / initE) > EDRIFTMAX:
            deltaPercent = '{:04f}%'.format((finalE - initE) / initE * 100)
            sys.exit('Energy drift of ' + deltaPercent + ' greater than {:04f}%'.format(EDRIFTMAX * 100))
        else:
            print(f'initE : {initE}')
            print(f'finalE: {finalE}')
            print('Energy % change: {:04f}%'.format((finalE - initE) / initE * 100))

    #
    # draw the path
    draw = True
    if draw:
        # Make an image every di time points, corresponding to a frame rate of fps frames per second.
        fps = 30
        di = int(1/fps/dt)
        fig = plt.figure(figsize=(8.3333, 6.25))    # dpi=72
        ax = fig.add_subplot(111)

        # Bob circle radius
        # Rods are "l" long (typically 1). Circles must be limited to fraction of rod length to avoid overlap
        r = sum(l) / 20  # 0.05 - 0.1
        r = min(0.25, r)

        # Fade time of bob trails
        trail_secs = 1.5
        maxTrail = int(trail_secs / dt)

        # x and y coords for each bob at all frames
        xs, ys = xy_coords(y)

        #
        # Drawing frames
        maxFrames = timeSteps // di
        print(f'\nDRAWING {maxFrames} FRAMES\n')
        for i in range(0, timeSteps, di):
            if (i // di) % 100 == 0:
                print(i // di, '/', maxFrames)
            make_plot(xs, ys, i, r, maxTrail, maxFrames)
        drawFinishTime = perf_counter()

        #
        # Animating frames with ffmpeg
        print('\nCONVERTING IMAGES TO GIF\n')
        os.chdir(imgFolder)
        if os.path.exists("pend.gif"):
            os.remove("pend.gif")

        trueFPS = int((timeSteps // di) / tmax)
        trueTime = (timeSteps // di) / trueFPS
        gifName = "pend.gif"
        # have to make a palette from the color palette of the png frames
        os.system("ffmpeg -hide_banner -loglevel error -i _img%04d.png -vf palettegen palette.png")
        # now animate the frames and save as gifName
        os.system("ffmpeg -hide_banner -loglevel error -framerate " + str(trueFPS) +
                  " -i _img%04d.png -i palette.png -lavfi paletteuse " + gifName)

        # remove frames after done animating them
        files = os.listdir(os.getcwd())
        for file in files:
            if file.endswith(".png"):
                os.remove(file)

        # print stats
        print(f'DONE: Saved {trueTime:.2f}s, {trueFPS} FPS gif \"{gifName}\" to {os.getcwd()} \\ {imgFolder}\n')
        print(f'TIME ELAPSED:       {(perf_counter() - startTime):.1f}s')
        print(f'TIME CALCULATING:   {(calcFinishTime - startTime):.1f}s')
        print(f'TIME DRAWING:       {(drawFinishTime - calcFinishTime):.1f}s')
        print(f'TIME ANIMATING:     {(perf_counter() - drawFinishTime):.1f}s')
