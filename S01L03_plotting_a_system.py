"""
Plotting a System of Linear Equations
https://www.udemy.com/course/machine-learning-data-science-foundations-masterclass/learn/lecture/23006968#overview

Section 01 Lesson 03

04 January 2024

https://github.com/jonkrohn/ML-foundations

"""
# pip install numpy
# pip install matplotlib
import matplotlib.pyplot as plt
import numpy as np

start_mins = 0
finish_mins = 40
num_points = 1000
time_steps = np.linspace(start_mins, finish_mins, num_points) # start, finish, n points

distance_robber = 2.5 * time_steps
distance_sheriff = 3 * (time_steps - 5)

"""
matplotlib.pyplot.subplots
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
Example down below

Create a figure and a set of subplots.
This utility wrapper makes it convenient to create common layouts of subplots, 
including the enclosing figure object, in a single call.

matplotlib.pyplot.subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, 
    squeeze=True, width_ratios=None, height_ratios=None, subplot_kw=None, gridspec_kw=None, **fig_kw)

"""
# fig is for everything to do with the "figure" - called via plt
# sub1 is the first sub-plot: what is plotted "on" the figure
fig, sub1 = plt.subplots()

# Title and label axes
plt.title('A Bank Robber Caught')
plt.xlabel('time (in minutes)')
plt.ylabel('distance (in km)')

# The range to plot over
sub1.set_xlim([0, 40])
sub1.set_ylim([0, 100])

# Do the plot
sub1.plot(time_steps, distance_robber, c='green')
sub1.plot(time_steps, distance_sheriff, c='brown')

# Cheat because he know the solution
plt.axvline(x=30, color='purple', linestyle='--') 
plt.axhline(y=75, color='purple', linestyle='--')

plt.show()
###################################################################




















